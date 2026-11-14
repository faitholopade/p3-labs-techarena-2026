"""Test fixed narrow-slate transfers into V51 remaining base/memory branches."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from calibration_model import predict
from rebuild_benchmark import sha, write
from research_v45 import mix as v45_mix
from research_v43 import WEIGHTS, MASS


def correction(control, plain, recipe):
    return sum(.5 * WEIGHTS[name] * recipe[name] * (plain - control[name]) for name in ['base', 'memory'])


def screen(args, plan):
    baseline_data = dict(np.load(args.v51 / 'selected_evidence.npz'))
    controls = {day: dict(np.load(args.benchmark / 'predictions' / (day + '.npz')))
                for day in ['weekend', 'thursday', 'friday']}
    narrow = dict(np.load(args.v52 / 'folds/predictions.npz'))
    current = dict(np.load(args.v51 / 'current_predictions.npz'))
    baseline = baseline_data['candidate']
    assert np.array_equal(baseline, current['weekend'])
    assert abs(roc_auc_score(baseline_data['labels'], baseline) - .8378178361843427) < 1e-12
    labels = baseline_data['labels']
    np.savez_compressed(args.out / 'v51_baseline.npz', indices=baseline_data['indices'], labels=labels,
                        users=baseline_data['users'], period=baseline_data['period'], candidate=baseline)
    masks = {'combined': np.ones(len(labels), bool), 'earlier': baseline_data['period'] == 0,
             'later': baseline_data['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        candidate = baseline + correction(controls['weekend'], narrow['weekend'], recipe)
        assert np.isfinite(candidate).all() and candidate.min() >= 0
        assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(labels[mask], candidate[mask]) - roc_auc_score(labels[mask], baseline[mask]))
                 for name, mask in masks.items()}
        loss = float(log_loss(labels, candidate) - log_loss(labels, baseline))
        weekdays = {}
        for day in ['thursday', 'friday']:
            reference = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            old = current[day]
            changed = old + correction(controls[day], narrow[day], recipe) / MASS
            weekdays[day] = float(roc_auc_score(reference['labels'], changed) - roc_auc_score(reference['labels'], old))
        eligible = gains['combined'] >= .000010 and gains['earlier'] > 0 and gains['later'] > 0 and loss <= 0 and all(x > 0 for x in weekdays.values())
        rows.append({'recipe': recipe, 'gains': gains, 'logloss_change': loss,
                     'weekday_component_gains': weekdays, 'eligible': eligible})
        candidates[recipe['name']] = candidate
    valid = [row for row in rows if row['eligible']]; selected = None
    if valid:
        best = max(row['gains']['combined'] for row in valid)
        order = {recipe['name']: i for i, recipe in enumerate(plan['recipes'])}
        selected = min([row for row in valid if row['gains']['combined'] >= best - .000005],
                       key=lambda row: (row['recipe']['base'] + row['recipe']['memory'], row['recipe']['base'], order[row['recipe']['name']]))
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=baseline_data['indices'], labels=labels,
                            users=baseline_data['users'], period=baseline_data['period'], baseline=baseline,
                            candidate=candidates[selected['recipe']['name']])
    report = {'plan': plan, 'plan_sha256': sha(args.plan), 'source_sha256': sha(Path(__file__)),
              'prediction_inputs_sha256': {'slate': sha(args.v52 / 'folds/predictions.npz'), 'incumbent': sha(args.v51 / 'current_predictions.npz')},
              'baseline_auc': float(roc_auc_score(labels, baseline)), 'results': rows, 'selected': selected,
              'official_score': None, 'portal_uploads': 0}
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


def forward(args, plan):
    report = json.loads((args.out / 'report.json').read_text()); selected = report['selected']; assert selected
    reference = dict(np.load(args.v51 / 'forward_evidence.npz'))
    original = json.loads((args.v45 / 'forward_assessment/report.json').read_text())
    raw = np.load(args.benchmark / 'features/friday/raw.npy', mmap_mode='r')
    alignment = np.load(args.benchmark / 'features/friday/alignment.npz')
    pos = alignment['first_position']; assert np.array_equal(alignment['indices'][pos], reference['indices'])
    plain_path = args.v52 / 'forward/predictions.npz'; new = np.load(plain_path)
    assert np.array_equal(new['indices'], reference['indices'])
    control, hashes = {}, {}
    for name in ['base', 'memory']:
        path = args.controls / (name + '.pt'); hashes[name] = sha(path)
        assert hashes[name] == original['model_sha256'][name]
        model = torch.load(path, map_location='cpu', weights_only=False)
        assert model['train_events'] == 97717
        columns = 344 if name == 'base' else 388
        parts = []
        for start in range(0, len(pos), 1500):
            rows = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
            parts.append(predict(model, raw[rows, :columns]))
        control[name] = np.concatenate(parts)
    baseline = reference['candidate']
    assert abs(roc_auc_score(reference['labels'], baseline) - .8250131431384134) < 1e-12
    candidate = baseline + correction(control, new['prediction'], selected['recipe']) / MASS
    assert candidate.min() >= 0 and candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
    result = {'recipe': selected['recipe'], 'baseline': 'V51 normalized changed-component forward comparison',
              'baseline_auc': float(roc_auc_score(reference['labels'], baseline)),
              'gain': float(roc_auc_score(reference['labels'], candidate) - roc_auc_score(reference['labels'], baseline)),
              'logloss_change': float(log_loss(reference['labels'], candidate) - log_loss(reference['labels'], baseline)),
              'events': len(pos), 'control_sha256': hashes, 'plain_predictions_sha256': sha(plain_path),
              'source_sha256': sha(Path(__file__)), 'not_full_incumbent_validation': True,
              'limits': 'Preselected recipe; no retuning on this diagnostic. All factors are chronological. Omits unchanged V17 and V20 components.',
              'official_score': None, 'portal_uploads': 0}
    np.savez_compressed(args.out / 'forward_evidence.npz', indices=reference['indices'], labels=reference['labels'],
                        users=reference['users'], baseline=baseline, candidate=candidate)
    write(args.out / 'forward_report.json', result); print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'v51', 'v52', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--stage', choices=['screen', 'forward'], default='screen')
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v53_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if args.stage == 'screen': screen(args, plan)
    else:
        assert args.controls
        forward(args, plan)


if __name__ == '__main__': main()
