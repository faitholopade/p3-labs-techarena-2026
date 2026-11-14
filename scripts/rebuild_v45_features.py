"""Rebuild exact V45 extra-day features from the shared compact research bundle."""
import argparse
import gc
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from common import load_data, subset, binary_labels
from daily_inference import build
from history_tree_features import extend
from v25_exposure_memory import features as memory_features
from rebuild_benchmark import sha, write, runtime_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True, help='Restored research_v45 bundle directory.')
    args = parser.parse_args(); torch.set_num_threads(3)
    data, _, audit = load_data(runtime_path(args.runtime, 11) / 'data')
    reports = []
    for day in ['tuesday', 'wednesday']:
        folder = args.out / 'features' / day
        report = json.loads((folder / 'report.json').read_text())
        for name in ['alignment.npz', 'names.json']:
            assert sha(folder / name) == report['files'][name]
        if (folder / 'raw.npy').exists():
            assert sha(folder / 'raw.npy') == report['files']['raw.npy']
            reports.append({'day': day, 'exact': True, 'reused': True}); continue
        partial = folder / 'raw.partial.npy'
        if partial.exists():
            raise ValueError('Remove the incomplete derived matrix before restarting: ' + str(partial))
        alignment = dict(np.load(folder / 'alignment.npz')); indices = alignment['indices']
        cut = report['inputs']['cut']; factor = args.out / 'factors' / (day + '.pt')
        assert sha(factor) == report['factor_sha256']
        fit = subset(data, np.flatnonzero(data['timestamp'] < cut)); query = subset(data, indices)
        assert np.array_equal(binary_labels(query).ravel(), alignment['labels'])
        assert np.array_equal(query['user'], alignment['user'])
        query['target'] = np.full(len(indices), -1)
        assert fit['timestamp'].max() < cut <= query['timestamp'].min()
        assert query['timestamp'].max() < cut + 86400
        state = torch.load(factor, map_location='cpu', weights_only=False)
        assert state['cut'] == cut
        expected = json.loads((folder / 'names.json').read_text())
        raw, names = build(fit, query, cut, audit['n_items'], audit['n_users'], factor,
                           state_root=args.out / 'states', posterior_samples=256)
        memory, memory_names = memory_features(fit, query, state['state'], cut)
        destination = np.lib.format.open_memmap(partial, mode='w+', dtype=np.float32,
                                                shape=(len(indices) * 10, len(expected)))
        for start in range(0, len(indices), 4000):
            stop = min(start + 4000, len(indices))
            block, schema = extend(raw[start * 10:stop * 10], subset(query, np.arange(start, stop)),
                                   state['state'], names)
            assert schema + memory_names == expected
            destination[start * 10:stop * 10, :344] = block
            destination[start * 10:stop * 10, 344:] = memory[start * 10:stop * 10]
        destination.flush(); del destination, raw, memory, fit, query, state
        gc.collect()
        actual = sha(partial)
        assert actual == report['files']['raw.npy'], (day, actual)
        partial.rename(folder / 'raw.npy')
        reports.append({'day': day, 'exact': True, 'sha256': actual})
        print(reports[-1], flush=True)
    write(args.out / 'feature_reproduction.json', reports)


if __name__ == '__main__':
    main()
