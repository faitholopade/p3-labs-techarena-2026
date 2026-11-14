"""Append observed query context to the full base+memory feature schema."""
import numpy as np

NAMES = ['query_is_weekend'] + [f'query_context_{k}' for k in range(8)]


class ContextView:
    def __init__(self, raw, context):
        self.raw = raw
        c = np.asarray(context)
        assert c.ndim == 1 and np.isin(c, np.arange(8)).all()
        assert len(raw) == 10 * len(c)
        self.context = np.c_[(c // 4).astype(np.float32), np.eye(8, dtype=np.float32)[c.astype(int)]]
        self.shape = (len(raw), raw.shape[1] + len(NAMES))

    def __len__(self):
        return self.shape[0]

    def __getitem__(self, key):
        columns = None
        if isinstance(key, tuple):
            key, columns = key
        rows = np.arange(*key.indices(len(self))) if isinstance(key, slice) else np.asarray(key)
        result = np.concatenate([self.raw[key], self.context[rows // 10]], axis=-1)
        return result if columns is None else result[..., columns]


def augment(datasets, alignments, training_path):
    source = np.load(training_path)
    context, users, timestamps = [source[k] for k in ['context', 'user', 'timestamp']]
    summaries = []
    for data, alignment in zip(datasets, alignments):
        indices = alignment['indices']
        assert np.array_equal(users[indices], alignment['user'])
        if 'timestamp' in alignment:
            assert np.array_equal(timestamps[indices], alignment['timestamp'])
        else:
            # The archived weekend alignment stores period instead of timestamp.
            days = timestamps[indices].astype(np.int64) // 86400
            assert np.array_equal(days - days.min(), alignment['period'])
        c = context[indices]
        weekend = ((timestamps[indices].astype(np.int64) // 86400 + 3) % 7 >= 5)
        assert np.array_equal(c // 4, weekend)
        data['x'] = ContextView(data['x'], c)
        data['names'] = np.asarray(data['names'].tolist() + NAMES)
        summaries.append({'events': len(indices), 'context_counts': np.bincount(c.astype(int), minlength=8).tolist()})
    return summaries
