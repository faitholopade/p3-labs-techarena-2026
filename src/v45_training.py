"""V33's fixed training method with batches normalized from disk-backed arrays.

The six-day full fit otherwise allocates over 7 GB for normalized features.
Only feature storage changes: statistics, examples, RNG order and objective
match v33_training.train with no sample weights or memory masking.
"""
import copy
import gc
import time
import numpy as np
import torch
from torch import nn
from calibration_model import Calibrator


class Slates:
    def __init__(self, datasets, masks, selected, mean, std):
        self.datasets = datasets
        self.events = [np.flatnonzero(mask) for mask in masks]
        self.boundaries = np.r_[0, np.cumsum([len(x) for x in self.events])]
        self.selected = selected
        self.mean, self.std = mean, std

    def batch(self, events):
        result = np.empty((len(events), 10, len(self.selected)), np.float32)
        for group, data in enumerate(self.datasets):
            at = np.flatnonzero((events >= self.boundaries[group]) & (events < self.boundaries[group + 1]))
            if not len(at):
                continue
            source = self.events[group][events[at] - self.boundaries[group]]
            rows = (source[:, None] * 10 + np.arange(10)).ravel()
            raw = data['x'][rows][:, self.selected]
            raw = np.where(np.isfinite(raw), raw, self.mean)
            result[at] = np.clip((raw - self.mean) / self.std, -8, 8).reshape(-1, 10, len(self.selected))
        return torch.from_numpy(result.reshape(-1, len(self.selected)))


def train(ds, masks, seed, epochs=20, width=32, offset_name='posterior1.0_mix1.0_utility',
          rank_weight=.5, threads=3, architecture='plain'):
    torch.set_num_threads(threads); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    names = ds[0]['names'].tolist()
    assert all(data['names'].tolist() == names for data in ds)
    selected = list(range(70)) + list(range(140, len(names)))
    bad = {'item_context_delta', 'item_context_impressions_log', 'activity_log_observation_days',
           'activity_last_day_count', 'activity_last_context_count'}
    selected = [i for i in selected if names[i] not in bad]
    p = len(selected); total = np.zeros(p); sq = np.zeros(p); counts = np.zeros(p)
    n = sum(int(mask.sum()) for mask in masks) * 10
    for data, mask in zip(ds, masks):
        indices = np.flatnonzero(mask)
        for start in range(0, len(indices), 2000):
            rows = (indices[start:start + 2000, None] * 10 + np.arange(10)).ravel()
            values = data['x'][rows][:, selected].astype(float)
            finite = np.isfinite(values); counts += finite.sum(0); values[~finite] = 0
            total += values.sum(0); sq += (values * values).sum(0)
    mu = total / np.maximum(counts, 1)
    sd = np.sqrt(np.maximum(sq / np.maximum(counts, 1) - mu ** 2, 0)); keep = sd > 1e-6
    selected = np.array(selected)[keep].tolist(); mu = mu[keep].astype(np.float32)
    sd = np.maximum(sd[keep], 1e-3).astype(np.float32); ncol = len(selected)
    offset = torch.empty(n); labels = np.empty(n, np.float32)
    position = 0; offset_column = names.index(offset_name)
    for data, mask in zip(ds, masks):
        indices = np.flatnonzero(mask)
        for start in range(0, len(indices), 2000):
            rows = (indices[start:start + 2000, None] * 10 + np.arange(10)).ravel()
            offset[position:position + len(rows)] = torch.from_numpy(data['x'][rows, offset_column].copy())
            labels[position:position + len(rows)] = data['labels'][rows]
            position += len(rows)
    del values
    gc.collect()
    features = Slates(ds, masks, selected, mu, sd)
    target = np.where(labels.reshape(-1, 10).sum(1) > 0, labels.reshape(-1, 10).argmax(1), 10)
    nevent = len(target); net = Calibrator(ncol, width, architecture)
    avg = copy.deepcopy(net); navg = 0
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.1); curve = []
    print({'training_events': nevent, 'features': ncol, 'seed': seed, 'disk_backed_batches': True}, flush=True)
    for epoch in range(1, epochs + 1):
        start = time.time(); net.train(); order = rng.permutation(nevent); losses = []
        for st in range(0, nevent, 1024):
            events = order[st:st + 1024]; rows = (events[:, None] * 10 + np.arange(10)).ravel()
            z = net(features.batch(events), offset[rows]).reshape(-1, 10)
            allz = torch.cat([z, torch.zeros(len(events), 1)], 1)
            loss = nn.functional.cross_entropy(allz, torch.tensor(target[events], dtype=torch.long))
            if rank_weight:
                probability = torch.softmax(allz, 1)[:, :10].ravel().clamp(1e-6, 1-1e-6)
                logits = torch.logit(probability); label = torch.tensor(labels[rows], dtype=torch.bool)
                positive = logits[label]; negative = logits[~label]
                if len(positive) and len(negative):
                    sampled = negative[torch.randint(len(negative), (len(positive), 32))]
                    loss += rank_weight * nn.functional.softplus(sampled - positive[:, None]).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5)
            opt.step(); losses.append(loss.item())
        if epoch >= max(1, epochs - 8):
            navg += 1
            with torch.no_grad():
                for destination, source in zip(avg.parameters(), net.parameters()):
                    destination.mul_((navg - 1) / navg).add_(source, alpha=1 / navg)
        curve.append(float(np.mean(losses)))
        if epoch % 5 == 0:
            print({'epoch': epoch, 'loss': curve[-1], 'seconds': time.time() - start}, flush=True)
    return {'state': avg.state_dict(), 'names': names, 'selected': selected, 'median': mu, 'mean': mu,
            'std': sd, 'clip': 8, 'architecture': architecture, 'width': width, 'offset_column': offset_column,
            'mode': 'event', 'seed': seed, 'epochs': epochs, 'rank_weight': rank_weight,
            'train_events': nevent, 'loss_curve': curve, 'memory_drop_probability': 0.,
            'memory_mask_columns': [j for j, i in enumerate(selected) if names[i].startswith('memory_')],
            'memory_mask_seed': seed + 33000, 'memory_mask_same_across_slate': True,
            'disk_backed_batches': True}
