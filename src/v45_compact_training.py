"""Exact in-memory slate storage: repeated event columns are stored once."""
import numpy as np
import torch
import v45_training


class CompactSlates:
    def __init__(self, datasets, masks, selected, mean, std):
        events = [np.flatnonzero(mask) for mask in masks]
        count = sum(len(x) for x in events)
        repeated = np.ones(len(selected), bool)
        for data, indices in zip(datasets, events):
            for start in range(0, len(indices), 2000):
                rows = (indices[start:start + 2000, None] * 10 + np.arange(10)).ravel()
                values = data['x'][rows][:, selected].reshape(-1, 10, len(selected))
                equal = (values == values[:, :1]) | (np.isnan(values) & np.isnan(values[:, :1]))
                repeated &= np.all(equal, axis=(0, 1))
        self.event_columns = np.flatnonzero(repeated)
        self.row_columns = np.flatnonzero(~repeated)
        self.columns = len(selected)
        self.row_values = np.empty((count * 10, len(self.row_columns)), np.float32)
        self.event_values = np.empty((count, len(self.event_columns)), np.float32)
        position = 0
        for data, indices in zip(datasets, events):
            for start in range(0, len(indices), 2000):
                chosen = indices[start:start + 2000]
                rows = (chosen[:, None] * 10 + np.arange(10)).ravel()
                values = data['x'][rows][:, selected]
                values = np.where(np.isfinite(values), values, mean)
                values = np.clip((values - mean) / std, -8, 8)
                self.row_values[position * 10:(position + len(chosen)) * 10] = values[:, self.row_columns]
                self.event_values[position:position + len(chosen)] = values[::10, self.event_columns]
                position += len(chosen)
        print({'compact_row_columns': len(self.row_columns), 'compact_event_columns': len(self.event_columns),
               'bytes': self.row_values.nbytes + self.event_values.nbytes, 'precision': 'float32, lossless'}, flush=True)

    def batch(self, events):
        rows = (events[:, None] * 10 + np.arange(10)).ravel()
        result = np.empty((len(events), 10, self.columns), np.float32)
        result[:, :, self.row_columns] = self.row_values[rows].reshape(len(events), 10, -1)
        result[:, :, self.event_columns] = self.event_values[events, None, :]
        return torch.from_numpy(result.reshape(-1, self.columns))


def train(*args, **kwargs):
    original = v45_training.Slates
    try:
        v45_training.Slates = CompactSlates
        model = v45_training.train(*args, **kwargs)
    finally:
        v45_training.Slates = original
    model['disk_backed_batches'] = False
    model['compact_repeated_event_columns'] = True
    return model
