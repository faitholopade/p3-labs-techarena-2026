"""Frozen six-day calibration fits and exact comparisons against scored V38."""
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
from sklearn.metrics import roc_auc_score, log_loss
from v45_training import train
from v33_training import train as dense_train
from calibration_model import predict
from rebuild_benchmark import sha, write, restore
from research_v43 import WEIGHTS, MASS, positions

LABELS = ['tuesday', 'wednesday', 'thursday', 'friday', 'weekend']


def datasets(args):
    datasets, alignments, hashes = [], [], {}
    for label in LABELS:
        folder = (args.out if label in ['tuesday', 'wednesday'] else args.benchmark) / 'features' / label
        raw = np.load(folder / 'raw.npy', mmap_mode='r')
        alignment = dict(np.load(folder / 'alignment.npz'))
        names = np.asarray(json.loads((folder / 'names.json').read_text()))
        assert raw.shape == (len(alignment['user']) * 10, len(names))
        datasets.append({'x': raw, 'labels': alignment['labels'], 'names': names})
        alignments.append(alignment)
        hashes[label] = {name: sha(folder / name) for name in ['raw.npy', 'alignment.npz', 'names.json']}
    for fold in range(3):
        used = np.unique(np.concatenate([a['user'][a['fold'] != fold] for a in alignments]))
        held = np.unique(np.concatenate([a['user'][a['fold'] == fold] for a in alignments]))
        assert not np.intersect1d(used, held).size
    return datasets, alignments, hashes


def fit(args, plan):
    ds, align, hashes = datasets(args)
    output = args.out / args.family / args.stage
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'plan_sha256': sha(args.plan), 'features': hashes, 'stage': args.stage, 'family': args.family,
              'sources': {name: sha(REPO / name) for name in ['src/v45_training.py',
                  'src/v33_training.py', 'src/calibration_model.py', 'scripts/research_v45.py']}}
    if (output / 'inputs.json').exists():
        assert json.loads((output / 'inputs.json').read_text()) == frozen
    write(output / 'inputs.json', frozen)
    predictions = {name: np.full(len(positions(al)) * 10, np.nan) for name, al in zip(LABELS[2:], align[2:])}
    records = []
    for fold in (range(3) if args.stage == 'folds' else [None]):
        keys = [0, 1, 2] if args.stage == 'forward' else list(range(5))
        masks = [align[k]['fold'] != fold if fold is not None else np.ones(len(align[k]['user']), bool) for k in keys]
        stem = f'fold_{fold}' if fold is not None else 'model'
        path, done = output / (stem + '.pt'), output / (stem + '_completed.json')
        if done.exists():
            record = json.loads(done.read_text()); assert sha(path) == record['sha256']
            model = torch.load(path, map_location='cpu', weights_only=False)
        else:
            if path.exists():
                raise ValueError('Unverified model: ' + str(path))
            started = time.time()
            method = dense_train if sum(int(mask.sum()) for mask in masks) <= 450000 else train
            model = method([ds[k] for k in keys], masks, seed=plan['seed'],
                          architecture='plain' if args.family == 'plain' else 'set', **plan['training'])
            temporary = path.with_suffix('.partial'); torch.save(model, temporary); temporary.rename(path)
            record = {'fold': fold, 'sha256': sha(path), 'training_events': model['train_events'],
                      'seconds': time.time() - started, 'users_disjoint': fold is not None}
            write(done, record)
        if args.stage == 'folds':
            for label, data, alignment in zip(LABELS[2:], ds[2:], align[2:]):
                pos = positions(alignment); test = np.flatnonzero(alignment['fold'][pos] == fold)
                for start in range(0, len(test), 1500):
                    events = test[start:start + 1500]
                    rows = (pos[events, None] * 10 + np.arange(10)).ravel()
                    dest = (events[:, None] * 10 + np.arange(10)).ravel()
                    predictions[label][dest] = predict(model, data['x'][rows])
        elif args.stage == 'forward':
            pos = positions(align[3]); blocks = []
            for start in range(0, len(pos), 1500):
                rows = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
                blocks.append(predict(model, ds[3]['x'][rows]))
            np.savez_compressed(output / 'predictions.npz', prediction=np.concatenate(blocks),
                                indices=align[3]['indices'][pos])
        records.append(record); print({'family': args.family, 'stage': args.stage, **record}, flush=True)
        del model; gc.collect()
    if args.stage == 'folds':
        assert all(np.isfinite(value).all() for value in predictions.values())
        np.savez_compressed(output / 'predictions.npz', **predictions)
    write(output / 'report.json', {'inputs': frozen, 'fits': records})


def mix(base, controls, new, recipe, mass=1.):
    mode = recipe['mode']; delta = 0.
    if mode == 'blend':
        weight = recipe['weight']; average = (new['plain'] + new['slate']) / 2
        return ((1 - weight) * mass * base + weight * average) / ((1 - weight) * mass + weight)
    if mode in ['replace_small', 'replace_both']:
        delta += WEIGHTS['small'] * (new['plain'] - controls['small'])
    if mode in ['replace_slate', 'replace_both']:
        delta += WEIGHTS['slate'] * (new['slate'] - controls['slate'])
    return base + delta / mass


def aggregate(args, plan, inputs):
    evidence = dict(np.load(inputs / 'evidence/matched_v38.npz'))
    baseline, labels = evidence['candidate'], evidence['labels']
    controls = {day: dict(np.load(args.benchmark / 'predictions' / f'{day}.npz')) for day in LABELS[2:]}
    predictions = {family: dict(np.load(args.out / family / 'folds/predictions.npz')) for family in plan['architectures']}
    masks = {'combined': np.ones(len(labels), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        new = {family: prediction['weekend'] for family, prediction in predictions.items()}
        candidate = mix(baseline, controls['weekend'], new, recipe)
        assert np.isfinite(candidate).all() and candidate.min() >= 0
        assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(labels[mask], candidate[mask]) - roc_auc_score(labels[mask], baseline[mask]))
                 for name, mask in masks.items()}
        loss = float(log_loss(labels, candidate) - log_loss(labels, baseline))
        weekdays = {}
        for day in LABELS[2:4]:
            reference = np.load(inputs / f'evidence/{day}_components.npz')
            new = {family: prediction[day] for family, prediction in predictions.items()}
            changed = mix(reference['candidate'], controls[day], new, recipe, MASS)
            weekdays[day] = float(roc_auc_score(reference['labels'], changed) - roc_auc_score(reference['labels'], reference['candidate']))
        eligible = gains['combined'] >= .000050 and gains['earlier'] > 0 and gains['later'] > 0 and loss <= 0 and all(x > 0 for x in weekdays.values())
        rows.append({'recipe': recipe, 'gains': gains, 'logloss_change': loss,
                     'weekday_component_gains': weekdays, 'eligible': eligible})
        candidates[recipe['name']] = candidate
    valid = [row for row in rows if row['eligible']]; selected = None
    if valid:
        best = max(row['gains']['combined'] for row in valid)
        order = {recipe['name']: i for i, recipe in enumerate(plan['recipes'])}
        selected = min([row for row in valid if row['gains']['combined'] >= best - .000005], key=lambda row: order[row['recipe']['name']])
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=labels,
                            users=evidence['users'], period=evidence['period'], baseline=baseline,
                            candidate=candidates[selected['recipe']['name']])
    report = {'plan': plan, 'plan_sha256': sha(args.plan), 'results': rows, 'selected': selected,
              'official_score': None, 'portal_uploads': 0}
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['folds', 'full', 'forward', 'aggregate'], required=True)
    parser.add_argument('--family', choices=['plain', 'slate'])
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v45_experiment_plan.json')
    args = parser.parse_args(); plan = json.loads(args.plan.read_text()); inputs, _ = restore(args.benchmark)
    if args.stage == 'aggregate':
        aggregate(args, plan, inputs)
    else:
        assert args.family
        fit(args, plan)


if __name__ == '__main__':
    main()
