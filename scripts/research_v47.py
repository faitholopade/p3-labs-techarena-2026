"""Frozen six-day tree comparison against officially scored V45."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from v47_tree import fit as tree_fit, predict
from rebuild_benchmark import sha, write
from research_v45 import datasets, mix as v45_mix
from research_v43 import WEIGHTS, MASS, positions


def mix(baseline, old, new, recipe, mass=1.):
    replaced = baseline + recipe['replacement'] * WEIGHTS['tree'] * (new - old) / mass
    w = recipe['blend']
    return ((1 - w) * mass * replaced + w * new) / ((1 - w) * mass + w)


def fit(args, plan):
    ds, align, hashes = datasets(SimpleNamespace(out=args.v45, benchmark=args.benchmark))
    output = args.out / args.stage
    output.mkdir(parents=True, exist_ok=True)
    frozen = {'plan_sha256': sha(args.plan), 'features': hashes, 'stage': args.stage,
              'sources': {name: sha(REPO / name) for name in ['src/v47_tree.py', 'scripts/research_v47.py']}}
    if (output / 'inputs.json').exists():
        assert json.loads((output / 'inputs.json').read_text()) == frozen
    write(output / 'inputs.json', frozen)
    predictions = {name: np.full(len(positions(a)) * 10, np.nan)
                   for name, a in zip(['thursday', 'friday', 'weekend'], align[2:])}
    records = []
    for fold in (range(3) if args.stage == 'folds' else [None]):
        keys = [0, 1, 2] if args.stage == 'forward' else list(range(5))
        masks = [align[k]['fold'] != fold if fold is not None else np.ones(len(align[k]['user']), bool) for k in keys]
        stem = f'fold_{fold}' if fold is not None else 'model'
        path, spec_path, done = output / (stem + '.txt'), output / (stem + '_spec.json'), output / (stem + '_completed.json')
        if done.exists():
            record = json.loads(done.read_text())
            assert sha(path) == record['sha256'] and sha(spec_path) == record['spec_sha256']
            model, spec = lgb.Booster(model_file=str(path)), json.loads(spec_path.read_text())
        else:
            if path.exists():
                raise ValueError('Unverified model: ' + str(path))
            start = time.time()
            model, spec = tree_fit([ds[k] for k in keys], masks, plan['training'])
            model.save_model(str(path)); write(spec_path, spec)
            record = {'fold': fold, 'sha256': sha(path), 'spec_sha256': sha(spec_path),
                      'training_events': spec['training_events'], 'seconds': time.time() - start,
                      'users_disjoint': fold is not None}
            write(done, record)
        if args.stage == 'folds':
            for label, data, alignment in zip(['thursday', 'friday', 'weekend'], ds[2:], align[2:]):
                pos = positions(alignment); test = np.flatnonzero(alignment['fold'][pos] == fold)
                for start in range(0, len(test), 1500):
                    events = test[start:start + 1500]
                    rows = (pos[events, None] * 10 + np.arange(10)).ravel()
                    dest = (events[:, None] * 10 + np.arange(10)).ravel()
                    predictions[label][dest] = predict(model, data['x'][rows], spec)
        elif args.stage == 'forward':
            pos = positions(align[3]); parts = []
            for start in range(0, len(pos), 1500):
                rows = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
                parts.append(predict(model, ds[3]['x'][rows], spec))
            np.savez_compressed(output / 'predictions.npz', prediction=np.concatenate(parts), indices=align[3]['indices'][pos])
        records.append(record); print(record, flush=True)
        del model; gc.collect()
    if args.stage == 'folds':
        assert all(np.isfinite(value).all() for value in predictions.values())
        np.savez_compressed(output / 'predictions.npz', **predictions)
    write(output / 'report.json', {'inputs': frozen, 'fits': records})


def screen(args, plan):
    evidence = dict(np.load(args.v45 / 'selected_evidence.npz'))
    baseline, labels = evidence['candidate'], evidence['labels']
    controls = {day: dict(np.load(args.benchmark / f'predictions/{day}.npz')) for day in ['weekend', 'thursday', 'friday']}
    new = dict(np.load(args.out / 'folds/predictions.npz'))
    heads = {name: dict(np.load(args.v45 / name / 'folds/predictions.npz')) for name in ['plain', 'slate']}
    masks = {'combined': np.ones(len(labels), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        candidate = mix(baseline, controls['weekend']['tree'], new['weekend'], recipe)
        assert candidate.min() >= 0 and candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(labels[m], candidate[m]) - roc_auc_score(labels[m], baseline[m])) for name, m in masks.items()}
        loss = float(log_loss(labels, candidate) - log_loss(labels, baseline))
        weekdays = {}
        for day in ['thursday', 'friday']:
            ref = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            old = v45_mix(ref['candidate'], controls[day], {k: v[day] for k, v in heads.items()}, {'mode': 'replace_both'}, MASS)
            changed = mix(old, controls[day]['tree'], new[day], recipe, MASS)
            weekdays[day] = float(roc_auc_score(ref['labels'], changed) - roc_auc_score(ref['labels'], old))
        eligible = gains['combined'] >= .000010 and min(gains.values()) > 0 and loss <= 0 and min(weekdays.values()) > 0
        rows.append(dict(recipe=recipe, gains=gains, logloss_change=loss, weekday_component_gains=weekdays, eligible=eligible))
        candidates[recipe['name']] = candidate
    valid = [row for row in rows if row['eligible']]; selected = None
    if valid:
        best = max(row['gains']['combined'] for row in valid)
        selected = next(row for row in rows if row['eligible'] and row['gains']['combined'] >= best - .000005)
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=labels,
                            users=evidence['users'], period=evidence['period'], baseline=baseline, candidate=candidates[selected['recipe']['name']])
    report = dict(plan=plan, plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)),
                  predictions_sha256=sha(args.out / 'folds/predictions.npz'), baseline_auc=float(roc_auc_score(labels, baseline)),
                  results=rows, selected=selected, official_score=None, portal_uploads=0)
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


def forward_score(args, plan):
    selected = json.loads((args.out / 'report.json').read_text())['selected']; assert selected
    ref = dict(np.load(args.v45 / 'forward_assessment/evidence.npz'))
    new = dict(np.load(args.out / 'forward/predictions.npz'))
    assert np.array_equal(new['indices'], ref['indices'])
    folder = args.benchmark / 'features/friday'
    raw = np.load(folder / 'raw.npy', mmap_mode='r'); alignment = np.load(folder / 'alignment.npz')
    pos = positions(alignment)
    assert np.array_equal(alignment['indices'][pos], ref['indices'])
    old_path = REPO / 'artifacts/v38/diagnostics/stable_tree.txt'
    spec = json.loads((old_path.parent / 'stable_tree_spec.json').read_text())
    model = lgb.Booster(model_file=str(old_path)); parts = []
    for start in range(0, len(pos), 1500):
        rr = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
        parts.append(predict(model, raw[rr], spec))
    baseline = ref['candidate']; candidate = mix(baseline, np.concatenate(parts), new['prediction'], selected['recipe'], MASS)
    report = dict(recipe=selected['recipe'], baseline_auc=float(roc_auc_score(ref['labels'], baseline)),
                  candidate_auc=float(roc_auc_score(ref['labels'], candidate)),
                  gain=float(roc_auc_score(ref['labels'], candidate) - roc_auc_score(ref['labels'], baseline)),
                  logloss_change=float(log_loss(ref['labels'], candidate) - log_loss(ref['labels'], baseline)),
                  events=len(pos), not_full_incumbent_validation=True, old_tree_sha256=sha(old_path),
                  predictions_sha256=sha(args.out / 'forward/predictions.npz'),
                  scope='Chronological Tuesday-Thursday tree versus Thursday-only tree; normalized changed components, unchanged V17/V20 omitted. No retuning.',
                  official_score=None, portal_uploads=0)
    np.savez_compressed(args.out / 'forward_evidence.npz', indices=ref['indices'], labels=ref['labels'], users=ref['users'], baseline=baseline, candidate=candidate)
    write(args.out / 'forward_report.json', report); print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['folds', 'forward', 'full', 'screen', 'forward-score'], required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v47_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if args.stage == 'screen': screen(args, plan)
    elif args.stage == 'forward-score': forward_score(args, plan)
    else: fit(args, plan)


if __name__ == '__main__': main()
