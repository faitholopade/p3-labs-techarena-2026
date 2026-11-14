"""Past-only user response gradients and bounded frozen-head corrections."""
import copy
import math
import time
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from uncertainty import torch_direction
from bounded_history import base_logits

FEATURE_NAMES = ['log_events', 'log_clicks'] + [f'{kind}_{axis}' for kind in
    ['mean_gradient', 'mean_curvature', 'newton25', 'newton100'] for axis in ['preference', 'exposure', 'intercept']]


@torch.no_grad()
def summarize(state, fit, cut):
    assert np.all(fit['timestamp'] < cut), 'Future outcomes in response features.'
    assert 'context_matrix' not in state, 'Context-dependent factors need explicit implementation.'
    nu = len(state['user.weight']); gradient = np.zeros((nu, 3)); fisher = np.zeros((nu, 3, 3))
    counts = np.bincount(fit['user'], minlength=nu)
    clicks = np.bincount(fit['user'], weights=fit['target'] < 10, minlength=nu)
    iv = torch_direction(state['item.weight'], state['item_logstd.weight'].clamp(-3, 1).exp())
    uv = torch_direction(state['user.weight'], state['user_logstd.weight'].clamp(-3, 1).exp())
    beta = state['log_beta'].clamp(math.log(2), math.log(8)).exp()
    eb = state['log_exposure_beta'].clamp(math.log(.5), math.log(8)).exp()
    for start in range(0, len(fit['user']), 4096):
        sl = slice(start, start + 4096); users = fit['user'][sl]; u = torch.tensor(users)
        c = torch.tensor(fit['items'][sl]); target = fit['target'][sl]
        pref = (iv[c] * uv[u, None]).sum(-1) * beta
        exposure = (state['exposure_items'][c] * state['exposure_user.weight'][u, None]).sum(-1) * eb / math.sqrt(8)
        z = pref + exposure + state['item_bias.weight'][c, 0] + state['outside.weight'][u]
        probability = torch.softmax(torch.cat([z, torch.zeros(len(u), 1)], 1), 1)[:, :10].double()
        basis = torch.stack([pref - pref.mean(1, keepdim=True), exposure - exposure.mean(1, keepdim=True), torch.ones_like(pref)], 2).double()
        expected = (basis * probability[:, :, None]).sum(1)
        actual = torch.zeros_like(expected); clicked = np.flatnonzero(target < 10)
        actual[clicked] = basis[clicked, target[clicked]]
        g = actual - expected
        h = torch.einsum('nk,nkd,nke->nde', probability, basis, basis) - expected[:, :, None] * expected[:, None, :]
        np.add.at(gradient, users, g.numpy()); np.add.at(fisher, users, h.numpy())
    denominator = counts[:, None] + 5
    blocks = [np.log1p(counts)[:, None], np.log1p(clicks)[:, None], gradient / denominator,
              np.diagonal(fisher, axis1=1, axis2=2) / denominator]
    for prior in [[25, 25, 4], [100, 100, 16]]:
        blocks.append(np.linalg.solve(fisher + np.diag(prior)[None], gradient[..., None])[..., 0])
    result = np.concatenate(blocks, axis=1).astype(np.float32)
    assert result.shape == (nu, len(FEATURE_NAMES)) and np.isfinite(result).all()
    assert np.all(result[counts == 0] == 0)
    return result


def query_basis(raw, names):
    pref = (raw[:, names.index('rx_user_raw')] - raw[:, names.index('rx_extra')]
            - raw[:, names.index('new_outside')]).reshape(-1, 10)
    exposure = (raw[:, names.index('rx_extra')] - raw[:, names.index('rx_item_bias')]).reshape(-1, 10)
    result = np.stack([pref - pref.mean(1, keepdims=True), exposure - exposure.mean(1, keepdims=True), np.ones_like(pref)], 2)
    assert np.isfinite(result).all()
    return result.astype(np.float32)


class ResponseCorrection(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(14, 16), nn.SiLU(), nn.Dropout(.1), nn.Linear(16, 3))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)

    def forward(self, features, basis):
        coefficient = torch.tanh(self.net(features)) * features.new_tensor([.25, .25, .5])
        return .5 * torch.tanh((basis * coefficient[:, None]).sum(2))


def fit(base, datasets, masks, seed=20261029, epochs=8, threads=3, rank_weight=.5,
        checkpoint_path=None, epoch_callback=None):
    torch.set_num_threads(threads); torch.manual_seed(seed); rng = np.random.default_rng(seed)
    utilities = np.concatenate([d['utility'][m] for d, m in zip(datasets, masks)])
    labels = np.concatenate([d['labels'].reshape(-1, 10)[m] for d, m in zip(datasets, masks)])
    features = np.concatenate([d['response'][m] for d, m in zip(datasets, masks)])
    basis = np.concatenate([d['basis'][m] for d, m in zip(datasets, masks)])
    mean = features.mean(0, dtype=np.float64).astype(np.float32)
    std = np.maximum(features.std(0, dtype=np.float64), 1e-3).astype(np.float32)
    features = torch.from_numpy(np.clip((features - mean) / std, -8, 8)); basis = torch.from_numpy(basis)
    target = np.where(labels.sum(1) > 0, labels.argmax(1), 10)
    net = ResponseCorrection(); averaged = copy.deepcopy(net); navg = 0
    optimizer = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.1); curve = []; completed_epoch = 0
    config = dict(seed=seed, epochs=epochs, threads=threads, rank_weight=rank_weight, events=len(target))
    if checkpoint_path is not None and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False); assert saved['config'] == config
        net.load_state_dict(saved['net']); averaged.load_state_dict(saved['average']); optimizer.load_state_dict(saved['optimizer'])
        navg = saved['navg']; curve = saved['curve']; completed_epoch = saved['epoch']
        rng.bit_generator.state = saved['numpy_rng']; torch.set_rng_state(saved['torch_rng'])
    for epoch in range(completed_epoch + 1, epochs + 1):
        started = time.time(); net.train(); order = rng.permutation(len(target)); losses = []
        for start in range(0, len(target), 1024):
            ev = order[start:start + 1024]
            z = torch.from_numpy(utilities[ev]) + net(features[ev], basis[ev])
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
                for dst, src in zip(averaged.parameters(), net.parameters()): dst.mul_((navg-1)/navg).add_(src, alpha=1/navg)
        entry = dict(epoch=epoch, seconds=time.time()-started, loss=float(np.mean(losses)))
        curve.append(entry); print(entry, flush=True)
        if checkpoint_path is not None:
            temporary = checkpoint_path.with_suffix('.partial')
            torch.save(dict(config=config, epoch=epoch, net=net.state_dict(), average=averaged.state_dict(),
                            optimizer=optimizer.state_dict(), navg=navg, curve=curve,
                            numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), temporary)
            temporary.replace(checkpoint_path)
        if epoch_callback is not None: epoch_callback(epoch)
    return dict(base_spec=base, state=averaged.state_dict(), response_names=FEATURE_NAMES, response_mean=mean,
                response_std=std, names=base['names'], architecture='bounded_user_response', width=base['width'],
                train_events=len(target), seed=seed, epochs=epochs, memory_drop_probability=0., base_frozen=True,
                rank_weight=rank_weight, curve=curve)


@torch.no_grad()
def predict(model, raw, response, basis=None):
    torch.set_num_threads(3); net = ResponseCorrection(); net.load_state_dict(model['state']); net.eval()
    utility = base_logits(model['base_spec'], raw)
    if basis is None: basis = query_basis(raw, model['names'])
    parts = []
    for start in range(0, len(response), 4096):
        sl = slice(start, start + 4096)
        features = np.clip((response[sl] - model['response_mean']) / model['response_std'], -8, 8)
        correction = net(torch.from_numpy(features), torch.from_numpy(basis[sl])).numpy()
        z = utility[sl] + correction
        parts.append(softmax(np.c_[z, np.zeros(len(z))], 1)[:, :10].ravel())
    return np.concatenate(parts)
