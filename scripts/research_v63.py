"""Fine-tune the saved six-day head on first daily user queries."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from calibration_model import predict
from first_query_adaptation import fit as train_adaptation
from rebuild_benchmark import sha, write, runtime_path
from research_v45 import datasets, LABELS
from research_v43 import MASS, positions


def fit(args, plan):
    assert not args.dense, 'This frozen-base implementation does not store dense normalized features.'
    torch.set_num_threads(plan['training']['threads'])
    ds, align, hashes = datasets(SimpleNamespace(out=args.v45, benchmark=args.benchmark))
    fold = args.fold if args.stage == 'folds' else None
    assert args.stage != 'folds' or fold in [0, 1, 2]
    stem = f'fold_{fold}' if fold is not None else 'model'
    base_path = args.v50 / args.stage / (stem + '.pt')
    base_key = f'artifacts/v50/research/{args.stage}/{stem}.pt'
    base_index = json.loads((REPO / 'artifacts/v50/research/index.json').read_text())
    expected = {m['path']: m['sha256'] for m in base_index['models']}
    assert sha(base_path) == expected[base_key]
    base = torch.load(base_path, map_location='cpu', weights_only=False)
    assert base['architecture'] == 'plain' and base['width'] == 16
    first_masks = []
    for alignment in align:
        mask = np.zeros(len(alignment['user']), bool)
        if 'period' in alignment:
            for period in np.unique(alignment['period']):
                local = np.flatnonzero(alignment['period'] == period)
                _, pos = np.unique(alignment['user'][local], return_index=True); mask[local[pos]] = True
        else:
            mask[alignment['first_position']] = True
        first_masks.append(mask)
    hashes['first_query_training_events'] = [int(m.sum()) for m in first_masks]
    output = args.out / args.stage; output.mkdir(parents=True, exist_ok=True)
    frozen = dict(plan_sha256=sha(args.plan), features=hashes, stage=args.stage, fold=fold,
                  base_model_sha256=sha(base_path), sources={name: sha(REPO / name) for name in [
                      'scripts/research_v63.py', 'scripts/research_v45.py', 'src/first_query_adaptation.py',
                      'src/calibration_model.py']})
    input_path = output / (stem + '_inputs.json')
    if input_path.exists(): assert json.loads(input_path.read_text()) == frozen
    write(input_path, frozen)
    path, done = output / (stem + '.pt'), output / (stem + '_completed.json')
    if done.exists():
        record = json.loads(done.read_text()); assert sha(path) == record['sha256']
        model = torch.load(path, map_location='cpu', weights_only=False)
    else:
        assert not path.exists(), 'Unverified model.'
        keys = [0, 1, 2] if args.stage == 'forward' else list(range(5))
        masks = [first_masks[k] & (align[k]['fold'] != fold) if fold is not None else first_masks[k] for k in keys]
        started = time.time()
        model = train_adaptation(base, [ds[k] for k in keys], masks, seed=plan['seed'],
                              epochs=plan['training']['epochs'], threads=plan['training']['threads'],
                              rank_weight=plan['training']['rank_weight'], checkpoint_path=output / (stem + '_resume.pt'))
        assert model['initial_normalization_preserved']
        temporary = path.with_suffix('.partial'); torch.save(model, temporary); temporary.rename(path)
        record = dict(fold=fold, sha256=sha(path), training_events=model['train_events'],
                      seconds=time.time() - started, users_disjoint=fold is not None, base_frozen=False)
        write(done, record)
    predictions = {}
    if args.stage == 'folds':
        for label, data, alignment in zip(LABELS[2:], ds[2:], align[2:]):
            pos = positions(alignment); test = np.flatnonzero(alignment['fold'][pos] == fold)
            predictions[label] = np.full(len(pos) * 10, np.nan)
            for start in range(0, len(test), 1500):
                events_local = test[start:start + 1500]; chosen = pos[events_local]
                rows = (chosen[:, None] * 10 + np.arange(10)).ravel()
                dest = (events_local[:, None] * 10 + np.arange(10)).ravel()
                predictions[label][dest] = predict(model, data['x'][rows])
        np.savez_compressed(output / (stem + '_predictions.npz'), **predictions)
    elif args.stage == 'forward':
        pos = positions(align[3]); parts = []
        for start in range(0, len(pos), 1500):
            chosen = pos[start:start + 1500]; rows = (chosen[:, None] * 10 + np.arange(10)).ravel()
            parts.append(predict(model, ds[3]['x'][rows]))
        np.savez_compressed(output / 'predictions.npz', prediction=np.concatenate(parts), indices=align[3]['indices'][pos])
    write(output / (stem + '_report.json'), dict(inputs=frozen, fit=record)); print(json.dumps(record), flush=True)

BRANCH_WEIGHTS = {'narrow': .095703125, 'plain': .109375, 'base': .0478515625, 'memory': .0478515625}


def correction(controls, new, recipe):
    return sum(weight * recipe[name] * (new - controls[name]) for name, weight in BRANCH_WEIGHTS.items())


def screen(args, plan):
    assert plan['branch_weights'] == BRANCH_WEIGHTS
    evidence = dict(np.load(args.v51 / 'selected_evidence.npz'))
    current = dict(np.load(args.v51 / 'current_predictions.npz'))
    old_narrow = dict(np.load(args.v51 / 'narrow_predictions.npz'))
    old_plain = dict(np.load(args.v45 / 'plain/folds/predictions.npz'))
    partials = [dict(np.load(args.out / f'folds/fold_{k}_predictions.npz')) for k in range(3)]
    new, controls = {}, {}
    for day in ['weekend', 'thursday', 'friday']:
        values = np.stack([p[day] for p in partials]); assert np.all(np.isfinite(values).sum(0) == 1)
        new[day] = np.nansum(values, axis=0)
        original = dict(np.load(args.benchmark / f'predictions/{day}.npz'))
        controls[day] = {'base': original['base'], 'memory': original['memory'],
                         'narrow': old_narrow[day], 'plain': old_plain[day]}
    np.savez_compressed(args.out / 'folds/predictions.npz', **new)
    baseline, y = evidence['candidate'], evidence['labels']
    assert np.array_equal(baseline, current['weekend'])
    assert abs(roc_auc_score(y, baseline) - .8378178361843427) < 1e-12
    masks = {'combined': np.ones(len(y), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        candidate = baseline + correction(controls['weekend'], new['weekend'], recipe)
        assert np.isfinite(candidate).all() and candidate.min() >= 0
        assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(y[m], candidate[m]) - roc_auc_score(y[m], baseline[m])) for name, m in masks.items()}
        loss = float(log_loss(y, candidate) - log_loss(y, baseline)); weekdays = {}
        for day in ['thursday', 'friday']:
            ref = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            changed = current[day] + correction(controls[day], new[day], recipe) / MASS
            weekdays[day] = float(roc_auc_score(ref['labels'], changed) - roc_auc_score(ref['labels'], current[day]))
        eligible = gains['combined'] >= .000010 and min(gains.values()) > 0 and loss <= 0 and min(weekdays.values()) > 0
        rows.append(dict(recipe=recipe, gains=gains, logloss_change=loss, weekday_component_gains=weekdays,
                         total_changed_weight=sum(BRANCH_WEIGHTS[k] * recipe[k] for k in BRANCH_WEIGHTS), eligible=eligible))
        candidates[recipe['name']] = candidate
    valid = [r for r in rows if r['eligible']]; selected = None
    if valid:
        best = max(r['gains']['combined'] for r in valid)
        selected = min([r for r in valid if r['gains']['combined'] >= best - .000005], key=lambda r: r['total_changed_weight'])
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=y, users=evidence['users'],
                            period=evidence['period'], baseline=baseline, candidate=candidates[selected['recipe']['name']])
    report = dict(plan=plan, plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)), baseline_auc=float(roc_auc_score(y, baseline)),
                  results=rows, selected=selected, official_score=None, portal_uploads=0)
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


def forward_score(args, plan):
    selected = json.loads((args.out / 'report.json').read_text())['selected']; assert selected
    ref = dict(np.load(args.v51 / 'forward_evidence.npz'))
    original = json.loads((args.v45 / 'forward_assessment/report.json').read_text())
    controls = {}
    for name, folder in [('plain', args.v45 / 'plain'), ('narrow', args.v50)]:
        saved = np.load(folder / 'forward/predictions.npz')
        assert np.array_equal(saved['indices'], ref['indices'])
        controls[name] = saved['prediction']
    raw = np.load(args.benchmark / 'features/friday/raw.npy', mmap_mode='r')
    alignment = np.load(args.benchmark / 'features/friday/alignment.npz')
    pos = alignment['first_position']; assert np.array_equal(alignment['indices'][pos], ref['indices'])
    for name in ['base', 'memory']:
        path = args.controls / (name + '.pt'); assert sha(path) == original['model_sha256'][name]
        model = torch.load(path, map_location='cpu', weights_only=False)
        columns = 344 if name == 'base' else 388
        parts = []
        for start in range(0, len(pos), 1500):
            rows = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
            parts.append(predict(model, raw[rows, :columns]))
        controls[name] = np.concatenate(parts)
    new = dict(np.load(args.out / 'forward/predictions.npz')); assert np.array_equal(new['indices'], ref['indices'])
    baseline = ref['candidate']; candidate = baseline + correction(controls, new['prediction'], selected['recipe']) / MASS
    report = dict(recipe=selected['recipe'], baseline_auc=float(roc_auc_score(ref['labels'], baseline)),
                  candidate_auc=float(roc_auc_score(ref['labels'], candidate)),
                  gain=float(roc_auc_score(ref['labels'], candidate) - roc_auc_score(ref['labels'], baseline)),
                  logloss_change=float(log_loss(ref['labels'], candidate) - log_loss(ref['labels'], baseline)),
                  events=len(pos), not_full_incumbent_validation=True,
                  scope='Preselected first-query adaptation recipe evaluated on first-Friday events. New, narrow and plain heads trained on all Tuesday-Thursday events; old base/memory Thursday only. Normalized changed components omit unchanged V17/V20. No retuning.',
                  official_score=None, portal_uploads=0)
    np.savez_compressed(args.out / 'forward_evidence.npz', indices=ref['indices'], labels=ref['labels'], users=ref['users'], baseline=baseline, candidate=candidate)
    write(args.out / 'forward_report.json', report); print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['runtime', 'benchmark', 'v45', 'v50', 'v51', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--controls', type=Path)
    parser.add_argument('--stage', choices=['folds', 'forward', 'full', 'screen', 'forward-score'], required=True)
    parser.add_argument('--fold', type=int, choices=[0, 1, 2]); parser.add_argument('--dense', action='store_true')
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v63_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    if args.stage == 'screen': screen(args, plan)
    elif args.stage == 'forward-score':
        assert args.controls
        forward_score(args, plan)
    else: fit(args, plan)


if __name__ == '__main__': main()

