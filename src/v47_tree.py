"""Same event tree as V38, constructed from bounded batches without a dense copy."""
import gc
import numbers
import time
import lightgbm as lgb
import numpy as np
from flexible_tree import probability
from v30_features import stable_columns
from v30_tree import predict


class ParityDataset(lgb.Dataset):
    def get_params(self):
        # Sequence construction calls get_params(), whereas ndarray construction
        # receives every parameter. Preserve the seed and prefilter threshold.
        result = super().get_params()
        for key in ['seed', 'min_data_in_leaf', 'num_threads', 'verbosity']:
            if self.params is not None and key in self.params:
                result[key] = self.params[key]
        return result


class EventSequence(lgb.Sequence):
    batch_size = 20000

    def __init__(self, raw, events, columns):
        self.raw, self.events, self.columns = raw, events, columns

    def __len__(self):
        return len(self.events) * 10

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            rows = self.events[key // 10] * 10 + key % 10
        else:
            idx = np.arange(*key.indices(len(self))) if isinstance(key, slice) else np.asarray(key)
            rows = self.events[idx // 10] * 10 + idx % 10
        block = np.asarray(self.raw[rows])[..., self.columns].astype(np.float64, copy=True)
        block[~np.isfinite(block)] = np.nan
        return np.ascontiguousarray(block)


def fit(datasets, masks, config):
    names = datasets[0]['names'].tolist()[:344]
    assert all(d['names'].tolist()[:344] == names for d in datasets)
    selected = stable_columns(names)
    offset_col = names.index('posterior1.0_mix1.0_utility')
    n = sum(int(m.sum()) for m in masks) * 10
    y, offset = np.empty(n, np.float64), np.empty(n, np.float64)
    sequences, pos = [], 0
    for ds, mask in zip(datasets, masks):
        events = np.flatnonzero(mask)
        if not len(events):
            continue
        sequences.append(EventSequence(ds['x'], events, selected))
        for start in range(0, len(events), 2000):
            rows = (events[start:start + 2000, None] * 10 + np.arange(10)).ravel()
            y[pos:pos + len(rows)] = ds['labels'][rows]
            offset[pos:pos + len(rows)] = ds['x'][rows, offset_col]
            pos += len(rows)
    assert pos == n
    if not np.isfinite(offset).all() or np.any(y.reshape(-1, 10).sum(1) > 1):
        raise ValueError('Invalid event labels or offsets.')
    first = True

    def objective(pred, data):
        nonlocal first
        if first:
            if not np.allclose(pred, offset, atol=1e-10, rtol=1e-10):
                raise ValueError('Initial event offset missing.')
            first = False
        p = probability(pred, 'event')
        return p - y, np.maximum(p * (1 - p), 1e-4)

    params = dict(objective=objective, metric='None', learning_rate=.025, num_leaves=7,
                  min_data_in_leaf=2000, lambda_l2=100., max_bin=127, feature_fraction=.85,
                  bagging_fraction=.85, bagging_freq=1, num_threads=config['threads'],
                  seed=config['seed'], verbosity=-1, force_col_wise=True, deterministic=True)
    started = time.time()
    data = ParityDataset(sequences, label=y, init_score=offset, free_raw_data=True,
                       params={k: v for k, v in params.items() if k != 'objective'})
    data.construct()
    del sequences
    gc.collect()
    print({'constructed_seconds': time.time() - started, 'events': n // 10}, flush=True)

    def progress(env):
        if (env.iteration + 1) % 100 == 0:
            print({'round': env.iteration + 1, 'seconds': time.time() - started}, flush=True)

    model = lgb.train(params, data, num_boost_round=config['rounds'], callbacks=[progress])
    return model, dict(names=names, selected=selected, offset_column=offset_col,
                       config=config, training_events=n // 10, mode='event')
