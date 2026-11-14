"""Fixed transfer of the remaining V51 tree branch to saved six-day neural heads."""
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


def mix(baseline, old, narrow, slate, recipe, mass=1.):
    new = narrow if recipe['head'] == 'narrow' else (narrow + slate) / 2
    return baseline + recipe['fraction'] * WEIGHTS['tree'] * (new - old) / mass


def screen(args, plan):
    evidence = dict(np.load(args.v51 / 'selected_evidence.npz'))
    baseline, labels = evidence['candidate'], evidence['labels']
    controls = {day: dict(np.load(args.benchmark / f'predictions/{day}.npz')) for day in ['weekend', 'thursday', 'friday']}
    new = dict(np.load(args.v51 / 'narrow_predictions.npz'))
    slate = dict(np.load(args.v45 / 'slate/folds/predictions.npz'))
    current = dict(np.load(args.v51 / 'current_predictions.npz'))
    assert np.array_equal(baseline, current['weekend'])
    masks = {'combined': np.ones(len(labels), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        candidate = mix(baseline, controls['weekend']['tree'], new['weekend'], slate['weekend'], recipe)
        assert candidate.min() >= 0 and candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(labels[m], candidate[m]) - roc_auc_score(labels[m], baseline[m])) for name, m in masks.items()}
        loss = float(log_loss(labels, candidate) - log_loss(labels, baseline))
        weekdays = {}
        for day in ['thursday', 'friday']:
            ref = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            old = current[day]
            changed = mix(old, controls[day]['tree'], new[day], slate[day], recipe, MASS)
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
                  predictions_sha256={'narrow': sha(args.v51 / 'narrow_predictions.npz'), 'slate': sha(args.v45 / 'slate/folds/predictions.npz')}, baseline_auc=float(roc_auc_score(labels, baseline)),
                  results=rows, selected=selected, official_score=None, portal_uploads=0)
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


def forward_score(args, plan):
    selected = json.loads((args.out / 'report.json').read_text())['selected']; assert selected
    ref = dict(np.load(args.v51 / 'forward_evidence.npz'))
    new = dict(np.load(args.v50 / 'forward/predictions.npz'))
    slate = dict(np.load(args.v45 / 'slate/forward/predictions.npz'))
    assert np.array_equal(slate['indices'], ref['indices'])
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
    baseline = ref['candidate']; candidate = mix(baseline, np.concatenate(parts), new['prediction'], slate['prediction'], selected['recipe'], MASS)
    report = dict(recipe=selected['recipe'], baseline_auc=float(roc_auc_score(ref['labels'], baseline)),
                  candidate_auc=float(roc_auc_score(ref['labels'], candidate)),
                  gain=float(roc_auc_score(ref['labels'], candidate) - roc_auc_score(ref['labels'], baseline)),
                  logloss_change=float(log_loss(ref['labels'], candidate) - log_loss(ref['labels'], baseline)),
                  events=len(pos), not_full_incumbent_validation=True, old_tree_sha256=sha(old_path),
                  predictions_sha256={'narrow': sha(args.v50 / 'forward/predictions.npz'), 'slate': sha(args.v45 / 'slate/forward/predictions.npz')},
                  scope='Chronological Tuesday-Thursday narrow/slate head versus Thursday-only tree; normalized changed components, unchanged V17/V20 omitted. No retuning.',
                  official_score=None, portal_uploads=0)
    np.savez_compressed(args.out / 'forward_evidence.npz', indices=ref['indices'], labels=ref['labels'], users=ref['users'], baseline=baseline, candidate=candidate)
    write(args.out / 'forward_report.json', report); print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'v50', 'v51', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--stage', choices=['screen', 'forward-score'], required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v56_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if args.stage == 'screen': screen(args, plan)
    elif args.stage == 'forward-score': forward_score(args, plan)



if __name__ == '__main__': main()
