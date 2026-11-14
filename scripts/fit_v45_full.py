"""Fit final six-day heads with lossless compact slate feature storage."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from research_v45 import datasets
from v45_compact_training import train
from v33_training import train as dense_train
from rebuild_benchmark import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--family', choices=['plain', 'slate'], required=True)
    parser.add_argument('--dense', action='store_true', help='Use the original dense trainer; reserve this for a single isolated full fit.')
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v45_experiment_plan.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text())
    data, alignment, hashes = datasets(args)
    output = args.out / args.family / 'full'; output.mkdir(parents=True, exist_ok=True)
    inputs = {'plan_sha256': sha(args.plan), 'feature_sha256': hashes, 'family': args.family, 'dense': args.dense,
              'sources': {name: sha(REPO / name) for name in ['scripts/fit_v45_full.py',
                  'src/v45_compact_training.py', 'src/v45_training.py', 'src/v33_training.py', 'src/calibration_model.py']}}
    if (output / 'inputs.json').exists():
        assert json.loads((output / 'inputs.json').read_text()) == inputs
    write(output / 'inputs.json', inputs)
    path = output / 'model.pt'; done = output / 'model_completed.json'
    if done.exists():
        assert sha(path) == json.loads(done.read_text())['sha256']; return
    assert not path.exists()
    started = time.time()
    method = dense_train if args.dense else train
    model = method(data, [np.ones(len(a['user']), bool) for a in alignment], seed=plan['seed'],
                  architecture='plain' if args.family == 'plain' else 'set', **plan['training'])
    temporary = path.with_suffix('.partial'); torch.save(model, temporary); temporary.rename(path)
    record = {'sha256': sha(path), 'training_events': model['train_events'], 'seconds': time.time() - started,
              'lossless_compact_storage': not args.dense, 'family': args.family}
    write(done, record); write(output / 'report.json', {'inputs': inputs, 'fits': [record]})
    print(record, flush=True)


if __name__ == '__main__':
    main()
