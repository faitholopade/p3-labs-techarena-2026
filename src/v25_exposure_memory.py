"""Past-only pair counts and recent shown-but-unclicked item similarities."""
import numpy as np
from repeats import rolling, NAMES
from uncertainty import numpy_direction


def features(fit, query, state, cut):
    if len(fit['user']) and fit['timestamp'].max() >= cut:
        raise ValueError('Snapshot contains a query-day outcome.')
    if np.any(query['timestamp'] < cut):
        raise ValueError('Query predates snapshot.')
    n = len(query['user']); ni = len(state['item.weight'])
    u = np.r_[fit['user'], query['user']]
    c = np.concatenate([fit['items'], query['items']])
    ts = np.repeat(np.r_[fit['timestamp'], query['timestamp']], 10)
    ctx = np.repeat(np.r_[fit['context'], query['context']], 10)
    label = np.r_[(np.arange(10)[None] == fit['target'][:, None]).ravel(),
                  np.full(n * 10, -1)]
    key = (u.astype(np.int64)[:, None] * ni + c).ravel()
    pair = rolling(key, ts, ctx, label, np.lexsort((ts, key)))[len(fit['user']) * 10:]
    del u, c, ts, ctx, label, key
    # Last five fully observed slates, frozen at the same cutoff as the factors.
    nu = len(state['user.weight'])
    recent = np.full((nu, 5), -1, dtype=np.int64)
    order = np.lexsort((np.arange(len(fit['user'])), fit['timestamp'], fit['user']))
    users = fit['user'][order]
    ends = np.r_[np.flatnonzero(users[1:] != users[:-1]) + 1, len(users)]
    starts = np.r_[0, ends[:-1]]
    for begin, end in zip(starts, ends):
        rows = order[max(begin, end - 5):end]
        recent[users[begin], -len(rows):] = rows
    pref = numpy_direction(state['item.weight'].numpy(),
                           state['item_logstd.weight'].clamp(-3, 1).exp().numpy())
    exp = state['exposure_items'].numpy().copy()
    exp /= np.maximum(np.linalg.norm(exp, axis=1, keepdims=True), 1e-8)
    pref[0] = 0; exp[0] = 0
    output = []; names = ['memory_pair_' + name for name in NAMES]
    for start in range(0, n, 512):
        stop = min(start + 512, n); past = recent[query['user'][start:stop]]
        valid = past >= 0; safe = np.maximum(past, 0)
        items = fit['items'][safe].copy(); items[~valid] = 0
        hit = np.arange(10)[None, None] == fit['target'][safe, None]
        click = hit & valid[:, :, None]; skip = (~hit) & valid[:, :, None]
        cols = []; local_names = []
        for family, vectors in [('pref', pref), ('exposure', exp)]:
            similarity = np.einsum('nkd,njsd->nkjs', vectors[query['items'][start:stop]], vectors[items])
            for window in [1, 3, 5]:
                for response, mask in [('clicked', click), ('skipped', skip)]:
                    m = mask[:, -window:].reshape(stop - start, -1)
                    sim = similarity[:, :, -window:].reshape(stop - start, 10, -1)
                    count = m.sum(1)[:, None]; mm = m[:, None]
                    mean = (sim * mm).sum(2) / np.maximum(count, 1)
                    maximum = np.where(count > 0, np.where(mm, sim, -1e9).max(2), 0)
                    std = np.sqrt(np.maximum((sim**2 * mm).sum(2) / np.maximum(count, 1) - mean**2, 0))
                    for suffix, value in [('mean', mean), ('max', maximum), ('std', std)]:
                        local_names.append(f'memory_{family}_{response}_last{window}_{suffix}')
                        cols.append(value.ravel())
        output.append(np.stack(cols, 1).astype(np.float32))
    result = np.c_[pair, np.concatenate(output)].astype(np.float32)
    names += local_names
    if not np.isfinite(result).all() or result.shape != (10*n, len(names)):
        raise ValueError('Invalid memory features.')
    return result, names
