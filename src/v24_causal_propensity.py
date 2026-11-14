"""Raw causal event-click propensity, independent of learned item factors.

Statistics are frozen at an explicit cutoff. Query targets must be masked;
query history may only contain clicks strictly before the query timestamp.
"""
import numpy as np
from scipy.special import logit, expit


def first_events(data, start, stop):
    ix = np.flatnonzero((data['timestamp'] >= start) & (data['timestamp'] < stop))
    order = np.argsort(data['timestamp'][ix], kind='stable')
    ix = ix[order]
    _, first = np.unique(data['user'][ix], return_index=True)
    return np.sort(ix[first])


def snapshot(fit, ni, nu, cut):
    if len(fit['user']) == 0 or np.max(fit['timestamp']) >= cut:
        raise ValueError('Snapshot must contain nonempty, strictly pre-cutoff data.')
    u, c, it = fit['user'], fit['context'], fit['items']
    hit = fit['target'] < 10
    yh = np.arange(10)[None, :] == fit['target'][:, None]
    g = float(np.mean(hit))
    out = {'cut': int(cut), 'max_timestamp': int(np.max(fit['timestamp'])), 'global': g}
    for name, ids, count, weights in [
        ('user', u, nu + 1, hit),
        ('user_hour', u * 4 + c % 4, (nu + 1) * 4, hit),
        ('user_day', u * 2 + c // 4, (nu + 1) * 2, hit),
        ('user_context', u * 8 + c, (nu + 1) * 8, hit),
        ('item', it.ravel(), ni + 1, yh.ravel()),
        ('item_hour', (it * 4 + c[:, None] % 4).ravel(), (ni + 1) * 4, yh.ravel()),
    ]:
        out[name + '_n'] = np.bincount(ids, minlength=count).astype(np.float32)
        out[name + '_h'] = np.bincount(ids, weights=weights, minlength=count).astype(np.float32)
    order = np.lexsort((fit['timestamp'], u))
    sorted_u = u[order]
    begin = np.r_[True, sorted_u[1:] != sorted_u[:-1]]
    end = np.r_[sorted_u[1:] != sorted_u[:-1], True]
    first, last = order[begin], order[end]
    for name in ['first_time', 'last_time', 'last_hit']:
        out[name] = np.zeros(nu + 1, dtype=np.float64)
    out['first_time'][u[first]] = fit['timestamp'][first]
    out['last_time'][u[last]] = fit['timestamp'][last]
    out['last_hit'][u[last]] = hit[last]
    return out


def features(query, state):
    if not np.all(query['target'] == -1):
        raise ValueError('Query targets must be masked.')
    if np.min(query['timestamp']) < state['cut']:
        raise ValueError('Query timestamps precede snapshot cutoff.')
    hmask = query['hist'] > 0
    if np.any(hmask & (query['hist_ts'] >= query['timestamp'][:, None])):
        raise ValueError('Query history is not strictly earlier than its event.')
    u, c, it, h = query['user'], query['context'], query['items'], query['hist']
    n = state['user_n'][u]
    hits = state['user_h'][u]
    g = state['global']
    baseline = (hits + 10 * g) / (n + 10)
    cols, names = [], []

    def add(name, value):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != u.shape or not np.isfinite(value).all():
            raise ValueError('Invalid feature ' + name)
        names.append(name)
        cols.append(value)

    def summaries(name, value):
        for suffix, fun in [('mean', np.mean), ('std', np.std), ('min', np.min), ('max', np.max)]:
            add(name + '_' + suffix, fun(value, axis=1))

    def logodds(value):
        return logit(np.clip(value, 1e-5, 1 - 1e-5))

    for k in [5, 10, 30, 100]:
        add('user_rate_logit_' + str(k), logodds((hits + k * g) / (n + k)))
    add('user_support_log', np.log1p(n))
    hour_id, day_id, ctx_id = u * 4 + c % 4, u * 2 + c // 4, u * 8 + c
    hn = state['user_hour_n'][hour_id]
    dn = state['user_day_n'][day_id]
    cn = state['user_context_n'][ctx_id]
    hour_rate = (state['user_hour_h'][hour_id] + 10 * baseline) / (hn + 10)
    day_rate = (state['user_day_h'][day_id] + 10 * baseline) / (dn + 10)
    ctx_rate = (state['user_context_h'][ctx_id] + 5 * hour_rate) / (cn + 5)
    add('user_hour_logit_delta', logodds(hour_rate) - logodds(baseline))
    add('user_day_logit_delta', logodds(day_rate) - logodds(baseline))
    add('user_context_logit_delta', logodds(ctx_rate) - logodds(hour_rate))
    add('hour_support_fraction', hn / np.maximum(n, 1))
    add('day_support_fraction', dn / np.maximum(n, 1))
    add('context_support_fraction', cn / np.maximum(n, 1))
    add('last_observed_event_hit', np.where(n > 0, state['last_hit'][u], -1))
    since = np.where(n > 0, np.maximum(query['timestamp'] - state['last_time'][u], 0) / 3600, 0)
    mean_gap = np.maximum((state['last_time'][u] - state['first_time'][u]) / 3600 / np.maximum(n - 1, 1), 1)
    add('last_observed_event_gap_log', np.log1p(since))
    add('relative_observed_gap_log', np.log1p(since / mean_gap))
    hour = query['timestamp'] % 86400 / 3600
    add('hour_sin', np.sin(2 * np.pi * hour / 24))
    add('hour_cos', np.cos(2 * np.pi * hour / 24))
    add('hour_bucket', c % 4)
    length = hmask.sum(1)
    add('history_length_scaled', length / 20)
    sh = np.sort(h, axis=1)
    unique = (sh[:, 0] > 0).astype(int) + ((sh[:, 1:] != sh[:, :-1]) & (sh[:, 1:] > 0)).sum(1)
    add('history_unique_scaled', unique / 20)
    age = np.maximum((query['timestamp'][:, None] - query['hist_ts']) / 3600, 0)
    newest = np.min(np.where(hmask, age, np.inf), axis=1)
    newest = np.where(length > 0, newest, 0)
    oldest = np.max(np.where(hmask, age, 0), axis=1)
    add('history_latest_age_log', np.log1p(newest))
    add('history_oldest_age_log', np.log1p(oldest))
    add('history_span_log', np.log1p(oldest - newest))
    for hours in [6, 24, 72, 168]:
        add('history_recent_fraction_' + str(hours), ((age < hours) & hmask).sum(1) / np.maximum(length, 1))
    item_n, item_h = state['item_n'], state['item_h']
    for k in [20, 100]:
        rate = (item_h[it] + k * g / 10) / (item_n[it] + k)
        summaries('slate_item_logit_' + str(k), logodds(rate))
    base_item_rate = (item_h[it] + 50 * g / 10) / (item_n[it] + 50)
    ids = it * 4 + (c[:, None] % 4)
    hour_item_rate = (state['item_hour_h'][ids] + 50 * base_item_rate) / (state['item_hour_n'][ids] + 50)
    summaries('slate_item_hour_logit_delta', logodds(hour_item_rate) - logodds(base_item_rate))
    hist_rate = (item_h[h] + 50 * g / 10) / (item_n[h] + 50)
    add('history_mean_item_rate', (hist_rate * hmask).sum(1) / np.maximum(length, 1))
    repeat = ((it[:, :, None] == h[:, None, :]) & hmask[:, None, :]).sum(2)
    add('slate_repeated_candidate_fraction', (repeat > 0).mean(1))
    add('slate_history_overlap_count', repeat.sum(1))
    return np.column_stack(cols).astype(np.float32), baseline.astype(np.float64), names


def predict(model, raw, baseline, threads=2):
    probability = expit(model.predict(raw, raw_score=True, num_threads=threads) + logit(np.clip(baseline, 1e-6, 1 - 1e-6)))
    if not np.isfinite(probability).all() or np.any((probability < 0) | (probability > 1)):
        raise ValueError('Invalid event probability.')
    return probability
