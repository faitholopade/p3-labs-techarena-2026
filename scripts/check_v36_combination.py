"""Two frozen combinations of the pooled tree and slate corrections against V33."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from backtest_v24_factor import sha256, write_json
from screen_v30_pooled import atomic_npz

TREE_WEIGHT = .07177734375
NEURAL_WEIGHT = .34375


def changed_component(old_tree, new_tree, h33, slate, recipe):
    base = (TREE_WEIGHT * old_tree + NEURAL_WEIGHT * h33) / (TREE_WEIGHT + NEURAL_WEIGHT)
    new = TREE_WEIGHT * new_tree + NEURAL_WEIGHT * h33
    mass = TREE_WEIGHT + NEURAL_WEIGHT
    if recipe == 'tree_plus_slate':
        new = .875 * new + .125 * slate
        mass = .875 * mass + .125
    return base, new / mass


def main():
    p = argparse.ArgumentParser(description=__doc__)
    keys = ['incumbent', 'v27-evidence', 'base-predictions', 'memory-weekend', 'memory-weekdays',
            'small-predictions', 'slate-predictions', 'old-tree', 'old-tree-weekdays', 'new-tree',
            'weekdays', 'tree-models', 'slate-models', 'plan', 'out']
    for key in keys:
        p.add_argument('--' + key, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())
    inputs = {key: sha256(getattr(a, key.replace('-', '_'))) for key in keys if key not in
              ['weekdays', 'tree-models', 'slate-models', 'out']}
    inputs['script_sha256'] = sha256(Path(__file__))
    checkpoints = []
    for folder, suffix in [(a.tree_models, '.txt'), (a.slate_models, '.pt')]:
        report = json.loads((folder / 'report.json').read_text())
        for fold in report['folds']:
            path = folder / (f"fold_{fold['fold']}" + suffix)
            assert sha256(path) == fold['model_sha256']
            checkpoints.append({'path': str(path), 'sha256': fold['model_sha256']})
    ev = dict(np.load(a.incumbent)); v27 = np.load(a.v27_evidence)
    for key in ['indices', 'labels', 'users', 'period']:
        assert np.array_equal(ev[key], v27[key])
    base = ev['candidate']; y = ev['labels']
    b = np.load(a.base_predictions); m_week = np.load(a.memory_weekend)
    m = dict(np.load(a.memory_weekdays)); m['weekend'] = m_week['prediction']
    assert np.array_equal(m_week['indices'], ev['indices'])
    small = np.load(a.small_predictions); slate = np.load(a.slate_predictions)
    new_tree = np.load(a.new_tree); old_weekend = np.load(a.old_tree)
    assert np.array_equal(old_weekend['indices'], ev['indices']) and np.array_equal(old_weekend['labels'], y)
    old_tree = dict(np.load(a.old_tree_weekdays)); old_tree['weekend'] = old_weekend['prediction']
    rebuilt = .65625 * v27['baseline'] + .109375 * b['weekend'] + .109375 * m['weekend'] + .125 * small['weekend']
    assert np.max(np.abs(rebuilt - base)) < 3e-16
    h33 = {label: (.109375 * b[label] + .109375 * m[label] + .125 * small[label]) / NEURAL_WEIGHT
           for label in ['thursday', 'friday', 'weekend']}
    masks = {'combined': np.ones(len(y), bool), 'earlier': ev['period'] == 0, 'later': ev['period'] == 1}
    auc = {name: float(roc_auc_score(y[mask], base[mask])) for name, mask in masks.items()}
    weekday_y = {}
    for label in ['thursday', 'friday']:
        al = np.load(a.weekdays / label / 'alignment.npz')
        rows = (al['first_position'][:, None] * 10 + np.arange(10)).ravel()
        weekday_y[label] = al['labels'][rows]
    results = []; candidates = {}
    tree_only = base + TREE_WEIGHT * (new_tree['weekend'] - old_tree['weekend'])
    for recipe in plan['recipes']:
        candidate = tree_only if recipe == 'tree_only' else .875 * tree_only + .125 * slate['weekend']
        assert np.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1
        assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(y[mask], candidate[mask]) - auc[name]) for name, mask in masks.items()}
        weekdays = {}
        for label in ['thursday', 'friday']:
            old, new = changed_component(old_tree[label], new_tree[label], h33[label], slate[label], recipe)
            weekdays[label] = {'baseline_auc': float(roc_auc_score(weekday_y[label], old)),
                               'candidate_auc': float(roc_auc_score(weekday_y[label], new)),
                               'gain': float(roc_auc_score(weekday_y[label], new) - roc_auc_score(weekday_y[label], old))}
        loss = float(log_loss(y, candidate) - log_loss(y, base))
        eligible = gains['earlier'] > 0 and gains['later'] > 0 and gains['combined'] >= .00001 and loss <= 0 and all(v['gain'] > 0 for v in weekdays.values())
        results.append({'recipe': recipe, 'gains': gains, 'logloss_change': loss,
                        'weekday_changed_component': weekdays, 'eligible_before_confirmation': eligible})
        candidates[recipe] = candidate
    eligible = [r for r in results if r['eligible_before_confirmation']]
    selected = None
    if eligible:
        best = max(r['gains']['combined'] for r in eligible)
        selected = min([r for r in eligible if r['gains']['combined'] >= best - .000001], key=lambda r: r['recipe'] != 'tree_only')
        atomic_npz(a.out / 'selected_evidence.npz', indices=ev['indices'], labels=y, users=ev['users'], period=ev['period'],
                   baseline=base, candidate=candidates[selected['recipe']])
    individual = {label: {'tree_gain': float(roc_auc_score(weekday_y[label], new_tree[label]) - roc_auc_score(weekday_y[label], old_tree[label])),
                           'slate_component_gain': float(roc_auc_score(weekday_y[label], (.875 * NEURAL_WEIGHT * h33[label] + .125 * slate[label]) / (.875 * NEURAL_WEIGHT + .125)) - roc_auc_score(weekday_y[label], h33[label]))}
                  for label in ['thursday', 'friday']}
    report = {'plan': plan, 'input_sha256': inputs, 'checkpoint_hashes': checkpoints, 'baseline_auc': auc,
              'results': results, 'individual_weekday_diagnostics': individual, 'selected': selected,
              'portal_uploads': 0, 'official_score': None}
    write_json(a.out / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
