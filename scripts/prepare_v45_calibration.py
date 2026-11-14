"""Build two additional chronological calibration days from fresh factors."""
import argparse
import gc
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from common import load_data, subset, binary_labels
from daily_inference import build
from history_tree_features import extend
from v25_exposure_memory import features as memory_features
from backtest_v24_factor import fit_command
from rebuild_benchmark import sha, write, runtime_path, restore


def prepare(args, plan):
    args.out.mkdir(parents=True, exist_ok=True)
    root = runtime_path(args.runtime, 11).resolve()
    folder = args.out / args.day
    folder.mkdir(parents=True, exist_ok=True)
    cut = plan['additional_days'][args.day]
    frozen = {'plan_sha256': sha(args.plan), 'data_sha256': sha(root / 'data/train.npz'),
              'cut': cut, 'sources': {name: sha(REPO / name) for name in [
                  'scripts/prepare_v45_calibration.py', 'src/train_factor_core.py',
                  'src/refine_factor_joint.py', 'src/refine_exposure_prior.py',
                  'src/projected.py', 'src/daily_inference.py', 'src/history_tree_features.py',
                  'src/v25_exposure_memory.py']}}
    if (folder / 'inputs.json').exists():
        assert json.loads((folder / 'inputs.json').read_text()) == frozen
    write(folder / 'inputs.json', frozen)
    fresh = folder / 'fresh'
    subprocess.run([sys.executable, str(REPO / 'scripts/backtest_v24_factor.py'),
                    '--root', str(root), '--out', str(fresh), '--cut', str(cut),
                    '--prepare-only'], check=True)
    basis = fresh / 'bases/exposure.npz'
    previous = fit_command([REPO / 'src/train_factor_core.py', '--root', root,
                            '--init', fresh / 'bases/p0_u0.5_i0.5_r5.npy', '--exposure', basis,
                            '--pool-cache', fresh / 'pools', '--cut', cut, '--out', folder / 'core',
                            '--seed', 3951, '--kl', .25, '--warm-weight', 1.], folder / 'core', 'residual.pt')
    for step in plan['factor_schedule']:
        command = [REPO / 'src' / step['script'], '--root', root, '--init', previous,
                   '--exposure', basis, '--pool-cache', fresh / 'pools', '--cut', cut,
                   '--out', folder / step['name'], '--threads', 2, '--epochs', step['epochs'],
                   '--lr', step['lr'], '--seed', step['seed']] + step['extra']
        old = previous
        previous = fit_command(command, folder / step['name'], 'residual.pt')
        if step['name'] == 'users':
            before = torch.load(old, map_location='cpu', weights_only=False)['state']
            after = torch.load(previous, map_location='cpu', weights_only=False)['state']
            for key in before:
                if key.split('.')[0] not in ['user', 'user_logstd', 'exposure_user', 'exposure_logstd', 'outside']:
                    assert torch.equal(before[key], after[key]), key
            del before, after
    output = args.out / 'features' / args.day
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'report.json').exists():
        report = json.loads((output / 'report.json').read_text())
        assert report['inputs'] == frozen
        for name in ['raw.npy', 'alignment.npz', 'names.json']:
            assert sha(output / name) == report['files'][name]
        print({'reused': args.day}, flush=True)
        return
    if (output / 'raw.partial.npy').exists() or (output / 'raw.npy').exists():
        raise ValueError('Unverified existing feature matrix.')
    inputs, _ = restore(args.benchmark)
    data, _, audit = load_data(root / 'data')
    fit = subset(data, np.flatnonzero(data['timestamp'] < cut))
    indices = np.flatnonzero((data['timestamp'] >= cut) & (data['timestamp'] < cut + 86400))
    query = subset(data, indices)
    labels = binary_labels(query).ravel()
    query['target'] = np.full(len(indices), -1)
    assert fit['timestamp'].max() < cut <= query['timestamp'].min()
    assert np.all((query['hist'] == 0) | (query['hist_ts'] < query['timestamp'][:, None]))
    assert np.all(np.diff(query['timestamp']) >= 0)
    _, first = np.unique(query['user'], return_index=True)
    first = np.sort(first)
    assert np.all((query['hist'][first] == 0) | (query['hist_ts'][first] < cut))
    del data
    assignment = (np.arange(audit['n_users'] + 1, dtype=np.int64) * 2654435761 % 3).astype(np.int8)
    observed = np.full(len(assignment), -1, np.int8)
    for label in ['thursday', 'friday', 'weekend']:
        alignment = np.load(inputs / f'alignment/{label}.npz')
        for fold in range(3):
            users = np.unique(alignment['user'][alignment['fold'] == fold])
            assert np.all((observed[users] == -1) | (observed[users] == fold))
            observed[users] = fold
            assignment[users] = fold
    bases = args.out / 'states' / str(cut)
    bases.mkdir(parents=True, exist_ok=True)
    for name in ['exposure.npz', 'p0_u0.5_i0.5_r5.npy']:
        destination = bases / name
        if not destination.exists():
            shutil.copyfile(fresh / 'bases' / name, destination)
        assert sha(destination) == sha(fresh / 'bases' / name)
    started = time.time()
    state = torch.load(previous, map_location='cpu', weights_only=False)
    assert state['cut'] == cut
    expected_names = json.loads((inputs / 'names.json').read_text())
    torch.set_num_threads(3)
    print({'stage': 'features', 'day': args.day, 'events': len(indices)}, flush=True)
    raw, names = build(fit, query, cut, audit['n_items'], audit['n_users'], previous,
                       state_root=args.out / 'states', posterior_samples=256)
    memory, memory_names = memory_features(fit, query, state['state'], cut)
    destination = np.lib.format.open_memmap(output / 'raw.partial.npy', mode='w+', dtype=np.float32,
                                            shape=(len(indices) * 10, len(expected_names)))
    for start in range(0, len(indices), 4000):
        stop = min(start + 4000, len(indices))
        block, schema = extend(raw[start * 10:stop * 10], subset(query, np.arange(start, stop)),
                               state['state'], names)
        assert schema + memory_names == expected_names
        destination[start * 10:stop * 10, :344] = block
        destination[start * 10:stop * 10, 344:] = memory[start * 10:stop * 10]
    destination.flush()
    del raw, memory, destination
    gc.collect()
    (output / 'raw.partial.npy').rename(output / 'raw.npy')
    np.savez_compressed(output / 'alignment.npz', indices=indices, labels=labels, user=query['user'],
                        timestamp=query['timestamp'], fold=assignment[query['user']], first_position=first)
    write(output / 'names.json', expected_names)
    report = {'inputs': frozen, 'events': len(indices), 'first_events': len(first), 'columns': len(expected_names),
              'factor_sha256': sha(previous), 'max_fit_timestamp': int(fit['timestamp'].max()),
              'query_targets_masked': True, 'first_query_history_before_cutoff': True,
              'all_histories_before_own_query': True, 'files': {name: sha(output / name)
                  for name in ['raw.npy', 'alignment.npz', 'names.json']}, 'seconds': time.time() - started}
    write(output / 'report.json', report)
    print({'complete': args.day, 'events': len(indices), 'feature_sha256': report['files']['raw.npy']}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime', 'benchmark', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--day', choices=['tuesday', 'wednesday'], required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v45_experiment_plan.json')
    args = parser.parse_args()
    args.runtime = args.runtime.resolve(); args.benchmark = args.benchmark.resolve(); args.out = args.out.resolve()
    prepare(args, json.loads(args.plan.read_text()))


if __name__ == '__main__':
    main()
