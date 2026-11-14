"""Refit V14 history and rate heads from chronological features and saved V9 heads."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import torch
from common import load_data, subset, binary_labels
from daily_inference import build
from sequence_calibration import history_pairs
from rate_training import fit
from rate_features import features as rate_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Extracted V11 runtime.')
    parser.add_argument('--v13-extension', required=True, help='Extracted V13 model extension with chronological feature states.')
    parser.add_argument('--cache', required=True, help='New directory for about 2.3 GB of features.')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root, extension = Path(args.root), Path(args.v13_extension)
    cache, out = Path(args.cache), Path(args.out)
    cache.mkdir(parents=True, exist_ok=False)
    out.mkdir(parents=True, exist_ok=True)
    data, _, audit = load_data(root / 'data')
    datasets = []
    for day, cutoff, period in [(25, 1737763200, 'earlier'), (26, 1737849600, 'later')]:
        indices = np.flatnonzero((data['timestamp'] >= cutoff) & (data['timestamp'] < cutoff + 86400))
        query = subset(data, indices)
        prior = subset(data, np.flatnonzero(data['timestamp'] < cutoff))
        factor = root / 'validation/v11/source_models' / f'{period}_joint24.pt'
        raw, names = build(prior, query, cutoff, audit['n_items'], audit['n_users'], factor,
                           state_root=extension / 'validation/source_states')
        filename = cache / f'day{day}.npy'
        np.save(filename, raw)
        rates = rate_features(prior, query, cutoff, audit['n_users'])
        del raw, prior
        gc.collect()
        state = torch.load(factor, map_location='cpu', weights_only=False)['state']
        pairs, mask = history_pairs(state, query)
        datasets.append({'x': np.load(filename, mmap_mode='r'),
                         'names': names, 'labels': binary_labels(query).ravel(),
                         'pairs': pairs, 'mask': mask, 'user': query['user'], 'rates': rates})
        print({'day': day, 'events': len(indices)}, flush=True)
    del data
    gc.collect()
    masks = [np.ones(len(day['user']), dtype=bool) for day in datasets]
    for seed in [731, 932, 1417]:
        initial = torch.load(root / 'models' / f'v9_calibration_{seed}.pt',
                             map_location='cpu', weights_only=False)
        if any(list(day['names']) != initial['names'] for day in datasets):
            raise ValueError('Chronological features differ from the calibration schema.')
        checkpoint = fit(initial, datasets, masks, seed, epochs=8, use_rates=True)
        torch.save(checkpoint, out / f'rate_{seed}.pt')
        print({'completed_seed': seed}, flush=True)


if __name__ == '__main__':
    main()
