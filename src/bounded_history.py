"""A bounded history correction with a completely frozen six-day base head."""
import copy
import time
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from calibration_model import Calibrator, normalized


class HistoryCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = nn.Sequential(nn.Linear(6, 12), nn.SiLU(), nn.Linear(12, 1))
        self.residual = nn.Sequential(nn.Linear(14, 16), nn.SiLU(), nn.Dropout(.1), nn.Linear(16, 1))
        for layer in [self.attention[-1], self.residual[-1]]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, pair, mask):
        pair = pair.float(); m = mask[:, None, :]; count = m.sum(2).clamp_min(1)
        logits = self.attention(pair).squeeze(-1).masked_fill(~m, -1e9)
        weight = torch.softmax(logits, 2) * m; summary = []
        for k in [0, 1]:
            score = pair[:, :, :, k]; mean = (score * m).sum(2) / count
            maximum = score.masked_fill(~m, -1e9).max(2).values
            maximum = torch.where(m.any(2), maximum, torch.zeros_like(maximum))
            recent = (score[:, :, -3:] * m[:, :, -3:]).sum(2) / m[:, :, -3:].sum(2).clamp_min(1)
            summary.extend([mean, (score * weight).sum(2) - mean, maximum,
                            ((score.square() * m).sum(2) / count - mean.square()).clamp_min(0),
                            recent - mean, score[:, :, -1] * m[:, :, -1] - mean])
        summary.extend([mask.sum(1)[:, None].expand(-1, 10) / 20, (pair[:, :, :, 2] * m).sum(2) / count])
        return .5 * torch.tanh(self.residual(torch.stack(summary, -1)).squeeze(-1))


@torch.no_grad()
def base_logits(base, raw):
    net = Calibrator(len(base['selected']), base['width'], base.get('architecture', 'plain'))
    net.load_state_dict(base['state']); net.eval(); parts = []
    for start in range(0, len(raw), 40960):
        values = raw[start:start + 40960]
        off = torch.from_numpy(np.asarray(values[:, base['offset_column']], np.float32).copy())
        parts.append(net(normalized(values, base), off).numpy())
    return np.concatenate(parts).reshape(-1, 10)


def fit(base, datasets, masks, seed=20261029, epochs=8, threads=3, rank_weight=.5,
        checkpoint_path=None, epoch_callback=None):
    torch.set_num_threads(threads); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    utilities = np.concatenate([d['utility'][mask] for d, mask in zip(datasets, masks)])
    labels = np.concatenate([d['labels'].reshape(-1, 10)[mask] for d, mask in zip(datasets, masks)])
    groups = np.concatenate([np.full(int(mask.sum()), k, np.int8) for k, mask in enumerate(masks)])
    local = np.concatenate([np.flatnonzero(mask) for mask in masks])
    target = np.where(labels.sum(1) > 0, labels.argmax(1), 10)
    net = HistoryCorrection(); averaged = copy.deepcopy(net); navg = 0
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.1); curve = []; completed_epoch = 0
    config = dict(seed=seed, epochs=epochs, threads=threads, rank_weight=rank_weight, events=len(target))
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        assert saved['config'] == config
        net.load_state_dict(saved['net']); averaged.load_state_dict(saved['average'])
        optimizer.load_state_dict(saved['optimizer']); navg = saved['navg']; curve = saved['curve']
        completed_epoch = saved['epoch']; rng.bit_generator.state = saved['numpy_rng']
        torch.set_rng_state(saved['torch_rng'])
        print({'resuming_after_epoch': completed_epoch}, flush=True)
    for epoch in range(completed_epoch + 1, epochs + 1):
        started = time.time(); net.train(); losses = []
        for start in range(0, len(target), 512):
            if start == 0: order = rng.permutation(len(target))
            ev = order[start:start + 512]
            pair = np.empty((len(ev), 10, 20, 6), np.float16); history_mask = np.empty((len(ev), 20), bool)
            for k, data in enumerate(datasets):
                at = np.flatnonzero(groups[ev] == k)
                pair[at] = data['pairs'][local[ev[at]]]; history_mask[at] = data['mask'][local[ev[at]]]
            z = torch.from_numpy(utilities[ev]) + net(torch.from_numpy(pair), torch.from_numpy(history_mask))
            allz = torch.cat([z, torch.zeros(len(ev), 1)], 1)
            loss = nn.functional.cross_entropy(allz, torch.tensor(target[ev], dtype=torch.long))
            probability = torch.softmax(allz, 1)[:, :10].ravel().clamp(1e-6, 1 - 1e-6)
            logits = torch.logit(probability); clicked = torch.tensor(labels[ev].ravel(), dtype=torch.bool)
            positive, negative = logits[clicked], logits[~clicked]
            if len(positive) and len(negative):
                sampled = negative[torch.randint(len(negative), (len(positive), 32))]
                loss += rank_weight * nn.functional.softplus(sampled - positive[:, None]).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 5)
            optimizer.step(); losses.append(loss.item())
        if epoch >= max(1, epochs - 2):
            navg += 1
            with torch.no_grad():
                for destination, source in zip(averaged.parameters(), net.parameters()):
                    destination.mul_((navg - 1) / navg).add_(source, alpha=1 / navg)
        entry = dict(epoch=epoch, seconds=time.time() - started, loss=float(np.mean(losses)))
        curve.append(entry); print(entry, flush=True)
        if checkpoint_path is not None:
            temporary = checkpoint_path.with_suffix('.partial')
            torch.save(dict(config=config, epoch=epoch, net=net.state_dict(), average=averaged.state_dict(),
                            optimizer=optimizer.state_dict(), navg=navg, curve=curve,
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), temporary)
            temporary.replace(checkpoint_path)
        if epoch_callback is not None:
            epoch_callback(epoch)
    return dict(base_spec=base, state=averaged.state_dict(), names=base['names'], architecture='bounded_history',
                width=base['width'], train_events=len(target), seed=seed, epochs=epochs,
                memory_drop_probability=0., base_frozen=True, rank_weight=rank_weight, curve=curve)


@torch.no_grad()
def predict(model, raw, pairs, mask):
    torch.set_num_threads(3); net = HistoryCorrection(); net.load_state_dict(model['state']); net.eval(); parts = []
    utility = base_logits(model['base_spec'], raw)
    for start in range(0, len(pairs), 1024):
        sl = slice(start, start + 1024)
        correction = net(torch.from_numpy(np.asarray(pairs[sl])), torch.from_numpy(np.asarray(mask[sl]))).numpy()
        z = utility[sl] + correction
        parts.append(softmax(np.c_[z, np.zeros(len(z))], axis=1)[:, :10].ravel())
    return np.concatenate(parts)
