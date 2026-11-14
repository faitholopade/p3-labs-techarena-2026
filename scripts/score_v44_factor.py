"""Score fixed V44 factors through V38's unchanged held-user heads."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from common import load_data, subset, binary_labels
from factor_ensemble import update_features
from history_tree_features import extend
from v25_exposure_memory import features as memory_features
from calibration_model import predict
from v30_tree import predict as tree_predict
from rebuild_benchmark import sha, write, restore, runtime_path

WEIGHTS = {'base': .095703125, 'memory': .095703125, 'small': .109375,
           'slate': .125, 'tree': .06280517578125}
MASS = sum(WEIGHTS.values())


def score(args, plan):
    inputs, metadata = restore(args.benchmark)
    cohort = args.cohort
    label = cohort if cohort in ['thursday', 'friday'] else 'weekend'
    folder = args.benchmark / 'features' / label
    alignment = dict(np.load(folder / 'alignment.npz'))
    positions = alignment.get('first_position', alignment.get('matched_position'))
    selected = np.arange(len(positions))
    if label == 'weekend':
        selected = np.flatnonzero(alignment['period'][positions] == (0 if cohort == 'earlier' else 1))
        positions = positions[selected]
    rows = (positions[:, None] * 10 + np.arange(10)).ravel()
    control_rows = (selected[:, None] * 10 + np.arange(10)).ravel()
    old = dict(np.load(args.benchmark / 'predictions' / f'{label}.npz'))
    ref = dict(np.load(inputs / ('evidence/matched_v38.npz' if label == 'weekend' else f'evidence/{label}_components.npz')))
    assert np.array_equal(ref['indices'][selected], alignment['indices'][positions])
    cut = plan['cutoffs'][cohort]
    if label == 'weekend':
        source_factor = runtime_path(args.runtime, 17) / f'validation/factors/{cohort}_factor_users_only.pt'
    else:
        source_factor = inputs / f'factors/{cohort}.pt'
    output = args.out / cohort / args.variant
    model_path = output / 'residual.pt'
    completed = json.loads((output / 'completed.json').read_text())
    assert sha(model_path) == completed['model_sha256']
    assert completed['inputs']['plan_sha256'] == sha(args.plan)
    assert completed['inputs']['factor_sha256'] == sha(source_factor)
    frozen = {'plan_sha256': sha(args.plan), 'factor_sha256': sha(model_path),
              'feature_sha256': sha(folder / 'raw.npy'), 'alignment_sha256': sha(folder / 'alignment.npz'),
              'source_sha256': {p: sha(REPO / p) for p in ['scripts/score_v44_factor.py', 'src/factor_ensemble.py',
                  'src/history_tree_features.py', 'src/v25_exposure_memory.py', 'src/calibration_model.py',
                  'src/v30_tree.py', 'src/posterior_inference.py']}}
    if (output / 'score_report.json').exists():
        report = json.loads((output / 'score_report.json').read_text())
        assert report['inputs'] == frozen and sha(output / 'predictions.npz') == report['prediction_sha256']
        print({'reused_score': str(output)}, flush=True)
        return
    started = time.time()
    data, _, _ = load_data(runtime_path(args.runtime, 11) / 'data')
    fit = subset(data, np.flatnonzero(data['timestamp'] < cut))
    query = subset(data, alignment['indices'][positions])
    labels = binary_labels(query).ravel()
    assert np.array_equal(labels, ref['labels'][control_rows])
    query['target'] = np.full(len(positions), -1)
    assert np.all((query['hist'] == 0) | (query['hist_ts'] < cut))
    assert fit['timestamp'].max() < cut <= query['timestamp'].min()
    del data
    raw = np.load(folder / 'raw.npy', mmap_mode='r')[rows].copy()
    names = json.loads((folder / 'names.json').read_text())
    original_state = torch.load(source_factor, map_location='cpu', weights_only=False)['state']
    test_positions = np.unique(np.linspace(0, len(positions) - 1, 64).astype(int))
    test_rows = (test_positions[:, None] * 10 + np.arange(10)).ravel()
    test_query = subset(query, test_positions)
    original = raw[test_rows, :284].copy()
    update_features(original, test_query, source_factor, posterior_samples=256)
    original, schema = extend(original, test_query, original_state, names[:284])
    assert schema == names[:344]
    original_memory, memory_names = memory_features(fit, test_query, original_state, cut)
    assert schema + memory_names == names
    original = np.c_[original, original_memory]
    parity = float(np.nanmax(abs(original - raw[test_rows])))
    assert np.allclose(original, raw[test_rows], rtol=0, atol=2e-6, equal_nan=True), parity
    del original, original_memory, original_state
    changed = raw[:, :284].copy()
    update_features(changed, query, model_path, posterior_samples=256)
    state = torch.load(model_path, map_location='cpu', weights_only=False)
    assert state['cut'] == cut
    changed, schema = extend(changed, query, state['state'], names[:284])
    memory, memory_names = memory_features(fit, query, state['state'], cut)
    changed = np.c_[changed, memory]
    assert schema + memory_names == names and changed.shape == raw.shape
    max_fit_timestamp = int(fit['timestamp'].max())
    del memory, fit, state
    gc.collect()
    parts = {name: np.full(len(rows), np.nan) for name in WEIGHTS}
    parity_errors = {}
    folds = alignment['fold'][positions]
    for fold in range(3):
        events = np.flatnonzero(folds == fold)
        sample_events = np.intersect1d(events, test_positions)
        sample_rows = (sample_events[:, None] * 10 + np.arange(10)).ravel()
        for name in parts:
            columns = 344 if name in ['base', 'tree'] else 388
            if name == 'tree':
                model = lgb.Booster(model_file=str(inputs / f'folds/tree/fold_{fold}.txt'))
                spec = json.loads((inputs / f'folds/tree/fold_{fold}_spec.json').read_text())
                method = lambda block: tree_predict(model, block, spec)
            else:
                model = torch.load(inputs / f'folds/{name}/fold_{fold}.pt', map_location='cpu', weights_only=False)
                method = lambda block: predict(model, block)
            if len(sample_rows):
                error = float(np.max(abs(method(raw[sample_rows, :columns]) - old[name][control_rows[sample_rows]])))
                assert error < 1e-7, (name, fold, error)
                parity_errors[f'{name}_{fold}'] = error
            for start in range(0, len(events), 1500):
                target_rows = (events[start:start + 1500, None] * 10 + np.arange(10)).ravel()
                parts[name][target_rows] = method(changed[target_rows, :columns])
    assert all(np.isfinite(value).all() for value in parts.values())
    delta = sum(weight * (parts[name] - old[name][control_rows]) for name, weight in WEIGHTS.items())
    baseline = ref['candidate'][control_rows]
    results = []
    for alpha in plan['recipes']:
        candidate = baseline + alpha * delta / (MASS if label != 'weekend' else 1)
        assert candidate.min() >= 0 and candidate.max() <= 1
        assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-7
        results.append({'alpha': alpha, 'baseline_auc': float(roc_auc_score(labels, baseline)),
                        'candidate_auc': float(roc_auc_score(labels, candidate)),
                        'gain': float(roc_auc_score(labels, candidate) - roc_auc_score(labels, baseline)),
                        'logloss_change': float(log_loss(labels, candidate) - log_loss(labels, baseline))})
    np.savez_compressed(output / 'predictions.npz', indices=alignment['indices'][positions], labels=labels,
                        users=query['user'].repeat(10), baseline=baseline, delta=delta, **parts)
    report = {'cohort': cohort, 'variant': args.variant, 'inputs': frozen, 'results': results,
              'events': len(positions), 'source_feature_max_abs_error': parity,
              'source_head_prediction_errors': parity_errors, 'query_targets_masked': True,
              'max_fit_timestamp': max_fit_timestamp, 'factor_cutoff': cut,
              'full_ensemble_comparison': label == 'weekend',
              'prediction_sha256': sha(output / 'predictions.npz'), 'seconds': time.time() - started,
              'portal_uploads': 0}
    write(output / 'score_report.json', report)
    print(json.dumps(report), flush=True)


def initial_screen(args, plan):
    reports = {v: json.loads((args.out / 'later' / v / 'score_report.json').read_text())
               for v in plan['variants']}
    candidates = reports['local75']['results']
    controls = reports['cluster_control']['results']
    eligible = [r['alpha'] for r, control in zip(candidates, controls)
                if r['alpha'] == control['alpha'] and r['gain'] >= .000020 and r['gain'] > control['gain']]
    report = {'plan_sha256': sha(args.plan), 'cohort': 'later', 'control': controls,
              'personalized': candidates, 'continue': bool(eligible), 'qualifying_alphas': eligible,
              'scope': 'Precommitted early screen, not a release decision.', 'portal_uploads': 0}
    write(args.out / 'initial_screen.json', report)
    print(json.dumps(report), flush=True)


def aggregate(args, plan):
    evidence = dict(np.load(args.benchmark / 'inputs/evidence/matched_v38.npz'))
    cohorts = ['earlier', 'later', 'thursday', 'friday']
    predictions = {variant: {cohort: dict(np.load(args.out / cohort / variant / 'predictions.npz'))
                             for cohort in cohorts} for variant in plan['variants']}
    baseline = evidence['candidate']; labels = evidence['labels']
    masks = {'combined': np.ones(len(labels), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    records = []; candidates = {}
    for alpha in plan['recipes']:
        by_variant = {}
        for variant, data in predictions.items():
            joined = {key: np.concatenate([data[c][key] for c in ['earlier', 'later']])
                      for key in ['indices', 'labels', 'users', 'baseline', 'delta']}
            for key in ['indices', 'labels', 'users']:
                assert np.array_equal(joined[key], evidence[key])
            assert np.array_equal(joined['baseline'], baseline)
            candidate = baseline + alpha * joined['delta']
            by_variant[variant] = candidate
        candidate = by_variant['local75']; control = by_variant['cluster_control']
        gains = {key: float(roc_auc_score(labels[mask], candidate[mask]) - roc_auc_score(labels[mask], baseline[mask]))
                 for key, mask in masks.items()}
        control_gain = float(roc_auc_score(labels, control) - roc_auc_score(labels, baseline))
        weekday = {}
        for cohort in ['thursday', 'friday']:
            d = predictions['local75'][cohort]
            weekday[cohort] = float(roc_auc_score(d['labels'], d['baseline'] + alpha / MASS * d['delta']) -
                                   roc_auc_score(d['labels'], d['baseline']))
        loss = float(log_loss(labels, candidate) - log_loss(labels, baseline))
        eligible = gains['combined'] >= .000050 and gains['earlier'] > 0 and gains['later'] > 0 and loss <= 0 and all(x > 0 for x in weekday.values()) and gains['combined'] > control_gain
        records.append({'alpha': alpha, 'gains': gains, 'control_combined_gain': control_gain,
                        'logloss_change': loss, 'weekday_component_gains': weekday, 'eligible': eligible})
        candidates[alpha] = candidate
    valid = [r for r in records if r['eligible']]; selected = None
    if valid:
        best = max(r['gains']['combined'] for r in valid)
        selected = min([r for r in valid if r['gains']['combined'] >= best - .000005], key=lambda r: r['alpha'])
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=labels,
                            users=evidence['users'], period=evidence['period'], baseline=baseline,
                            candidate=candidates[selected['alpha']])
    report = {'plan': plan, 'plan_sha256': sha(args.plan), 'results': records, 'selected': selected,
              'official_score': None, 'portal_uploads': 0}
    write(args.out / 'report.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime', 'benchmark', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--cohort', choices=['thursday', 'friday', 'earlier', 'later'])
    parser.add_argument('--variant', choices=['cluster_control', 'local75'])
    parser.add_argument('--initial-screen', action='store_true')
    parser.add_argument('--aggregate', action='store_true')
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v44_experiment_plan.json')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    if args.aggregate:
        aggregate(args, plan)
    elif args.initial_screen:
        initial_screen(args, plan)
    else:
        assert args.cohort and args.variant
        score(args, plan)


if __name__ == '__main__':
    main()
