"""Bounded per-item cyclic hour effects over a frozen chronological factor.

Training offsets are in-sample for the frozen factor, so training diagnostics
are not generalisation estimates. Only a subsequent chronological query set
can test whether the added effect transfers. Query outcomes are never needed.
"""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.special import softmax
import torch
from torch import nn

from common import subset
from posterior_inference import features as posterior_features


def posterior_probability(state, data, samples=256):
    """Use exactly the existing full-scale, pure-user Sobol posterior utility."""
    n = len(data['user'])
    result = np.empty((n, 10), np.float64)
    for start in range(0, n, 4096):
        rows = np.arange(start, min(start + 4096, n))
        q = subset(data, rows)
        q['target'] = np.full(len(rows), -1, np.int64)
        raw, names = posterior_features(state, q, samples=samples)
        utility = raw[:, names.index('posterior1.0_mix1.0_utility')].reshape(-1, 10)
        result[rows] = softmax(np.c_[utility, np.zeros(len(rows))], axis=1)[:, :10]
        if start % 32768 == 0:
            print(json.dumps({'stage': 'posterior_offsets', 'complete': int(rows[-1] + 1), 'events': n}), flush=True)
    return result


def clock(context):
    angle = (np.asarray(context) % 4) * (np.pi / 2)
    return np.c_[np.sin(angle), np.cos(angle)].astype(np.float32)


class ItemTime(nn.Module):
    def __init__(self, support):
        super().__init__()
        self.coefficient = nn.Embedding(len(support), 2, padding_idx=0)
        nn.init.zeros_(self.coefficient.weight)
        self.register_buffer('support', torch.tensor(support, dtype=torch.float32))

    def forward(self, items, time_basis):
        score = (self.coefficient(items) * time_basis[:, None, :]).sum(-1)
        return .5 * torch.tanh(score) * self.support[items]


def train(fit, probability, ni, out, factor_sha256, cut, threads=3):
    """Fixed 12-epoch conditional-choice fit; count shrinkage and Gaussian ridge."""
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    seed, epochs, prior_strength = 20261029, 12, 20.
    torch.set_num_threads(threads); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    clicked = np.flatnonzero(fit['target'] < 10)
    if len(clicked) == 0:
        raise ValueError('No clicked events to fit conditional item choice.')
    if probability.shape != (len(fit['user']), 10) or not np.isfinite(probability).all():
        raise ValueError('Invalid frozen factor probabilities.')
    counts = np.bincount(fit['items'].ravel(), minlength=ni + 1).astype(np.float32)
    support = counts / (counts + 100.)
    support[0] = 0
    model = ItemTime(support)
    averaged = copy.deepcopy(model)
    navg = 0
    opt = torch.optim.Adam(model.parameters(), lr=.01)
    basis = clock(fit['context'])
    offset = np.log(np.clip(probability, 1e-12, 1))
    curve = []
    for epoch in range(1, epochs + 1):
        started = time.monotonic()
        order = rng.permutation(clicked)
        loss_sum = 0.; n_seen = 0
        for start in range(0, len(order), 2048):
            ix = order[start:start + 2048]
            delta = model(torch.tensor(fit['items'][ix]), torch.tensor(basis[ix]))
            logits = torch.tensor(offset[ix], dtype=torch.float32) + delta
            loss = nn.functional.cross_entropy(logits, torch.tensor(fit['target'][ix]))
            loss = loss + prior_strength * model.coefficient.weight.square().sum() / len(clicked)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            loss_sum += float(loss.detach()) * len(ix); n_seen += len(ix)
        if epoch >= 9:
            navg += 1
            with torch.no_grad():
                for dst, src in zip(averaged.parameters(), model.parameters()):
                    dst.mul_((navg - 1) / navg).add_(src, alpha=1 / navg)
        rec = {'epoch': epoch, 'training_loss': loss_sum / n_seen, 'seconds': time.monotonic() - started}
        curve.append(rec); print(json.dumps({'stage': 'item_time_fit', **rec}), flush=True)
    spec = {'seed': seed, 'epochs': epochs, 'learning_rate': .01, 'prior_strength': prior_strength,
            'support_rule': 'pre-cutoff candidate impressions / (candidate impressions + 100)',
            'maximum_absolute_utility_correction': .5, 'average_epochs': [9, 10, 11, 12],
            'input': 'item ID and cyclic six-hour bucket; no weekend extrapolation',
            'objective': 'conditional clicked-item cross entropy',
            'probability_rule': 'preserve frozen factor event click mass',
            'offset_scope': 'Frozen factor posterior; training rows are in-sample for that factor.',
            'cut': int(cut), 'factor_sha256': factor_sha256,
            'training_events': len(fit['user']), 'clicked_training_events': len(clicked),
            'max_training_timestamp': int(fit['timestamp'].max()), 'curve': curve}
    ck = {'state': averaged.state_dict(), 'support': support, 'spec': spec}
    torch.save(ck, out / 'model.pt')
    (out / 'training.json').write_text(json.dumps(spec, indent=2) + '\n')
    return ck


@torch.no_grad()
def predict(ck, data, baseline_probability):
    model = ItemTime(ck['support'])
    model.load_state_dict(ck['state']); model.eval()
    result = np.empty_like(baseline_probability, dtype=np.float64)
    largest_delta = 0.
    for start in range(0, len(result), 4096):
        sl = slice(start, start + 4096)
        delta = model(torch.tensor(data['items'][sl]), torch.tensor(clock(data['context'][sl]))).numpy()
        largest_delta = max(largest_delta, float(np.max(np.abs(delta))))
        p = baseline_probability[sl]
        result[sl] = softmax(np.log(np.clip(p, 1e-12, 1)) + delta, axis=1) * p.sum(1, keepdims=True)
    if not np.isfinite(result).all() or not np.allclose(result.sum(1), baseline_probability.sum(1), atol=1e-12):
        raise ValueError('Conditional correction did not preserve event probability mass.')
    return result, {'max_absolute_utility_correction': largest_delta,
                    'max_event_mass_difference': float(np.max(np.abs(result.sum(1) - baseline_probability.sum(1))))}
