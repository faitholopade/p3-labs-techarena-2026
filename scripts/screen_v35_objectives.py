"""Frozen smaller-head ranking-objective comparison against scored V33."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'scripts'))
import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score
from daily_training import train
from calibration_model import predict
from screen_v29_pooled import load_datasets
from screen_v30_pooled import atomic_npz, position
from backtest_v24_factor import sha256, write_json


def predict_events(model, raw, events):
    result = []
    for start in range(0, len(events), 2000):
        rows = (events[start:start + 2000, None] * 10 + np.arange(10)).ravel()
        result.append(predict(model, raw[rows]))
    return np.concatenate(result)


def pooled_head(base, memory, small):
    return (.109375 * base + .109375 * memory + .125 * small) / .34375


def changed_head(base, memory, small, new, recipe):
    h = pooled_head(base, memory, small)
    if recipe['mode'] == 'replace_small_head':
        return (.109375 * base + .109375 * memory + .125 * new) / .34375
    if recipe['mode'] == 'replace_pooled_heads':
        return new
    if recipe['mode'] == 'blend':
        w = recipe['weight']
        return ((1 - w) * .34375 * h + w * new) / ((1 - w) * .34375 + w)
    raise ValueError('Unknown recipe')


def full_prediction(parent, base, memory, small, new, recipe):
    incumbent = .65625 * parent + .109375 * base + .109375 * memory + .125 * small
    if recipe['mode'] == 'replace_small_head':
        return .65625 * parent + .109375 * base + .109375 * memory + .125 * new
    if recipe['mode'] == 'replace_pooled_heads':
        return .65625 * parent + .34375 * new
    if recipe['mode'] == 'blend':
        return (1 - recipe['weight']) * incumbent + recipe['weight'] * new
    raise ValueError('Unknown recipe')


def fit(a, plan):
    config = plan['variants'][a.variant]
    ds, align = load_datasets(a, 388)
    out = a.out / a.variant
    if a.full:
        out = out / 'full'
    if a.forward:
        out = out / 'forward'
    out.mkdir(parents=True, exist_ok=True)
    inputs = {'config': config, 'plan_sha256': sha256(a.plan), 'incumbent_sha256': sha256(a.incumbent),
              'source_sha256': {str(path.relative_to(REPO)): sha256(path) for path in
                                [Path(__file__), REPO / 'src/daily_training.py', REPO / 'src/calibration_model.py']},
              'feature_reports': {str(folder): sha256(folder / 'report.json') for folder in
                                  [a.weekdays / 'thursday', a.weekdays / 'friday', a.weekends]},
              'alignment_sha256': [sha256(folder / 'alignment.npz') for folder in
                                    [a.weekdays / 'thursday', a.weekdays / 'friday', a.weekends]],
              'small_predictions_sha256': sha256(a.small_predictions),
              'full': a.full, 'forward': a.forward}
    if (out / 'plan.json').exists() and json.loads((out / 'plan.json').read_text()) != inputs:
        raise ValueError('Frozen run changed')
    write_json(out / 'plan.json', inputs)
    kw = dict(seed=config['seed'], epochs=20, width=config['width'], rank_weight=config['rank_weight'],
              threads=2, architecture=config['architecture'], offset_name='posterior1.0_mix1.0_utility')
    labels = ['thursday', 'friday', 'weekend']
    ev = np.load(a.incumbent)
    assert np.array_equal(align[2]['indices'][position(align[2])], ev['indices'])
    rows = (position(align[2])[:, None] * 10 + np.arange(10)).ravel()
    assert np.array_equal(align[2]['labels'][rows], ev['labels'])
    if a.full or a.forward:
        if (out / 'model.pt').exists():
            raise ValueError('Full/forward model already exists')
        keys = [0] if a.forward else [0, 1, 2]
        model = train([ds[k] for k in keys], [np.ones(len(align[k]['user']), bool) for k in keys], **kw)
        path = out / 'model.pt'; temporary = out / 'model.partial'
        torch.save(model, temporary); temporary.replace(path)
        report = {'plan': inputs, 'training_events': model['train_events'], 'model_sha256': sha256(path)}
        if a.forward:
            selection = json.loads((a.out / 'report.json').read_text())['selected']
            assert selection and selection['variant'] == a.variant
            recipe = {'mode': selection['mode'], 'weight': selection['weight']}
            pos = position(align[1]); rows = (pos[:, None] * 10 + np.arange(10)).ravel()
            y = align[1]['labels'][rows]
            forward = np.load(a.forward_controls / 'all_predictions.npz')
            controls = {}
            for name in ['base_20261029', 'memory_20261029']:
                completed = json.loads((a.forward_controls / (name + '_completed.json')).read_text())
                assert completed['model_sha256'] == sha256(a.forward_controls / (name + '.pt'))
                controls[name] = forward[name]
            new = predict_events(model, ds[1]['x'], pos)
            small_report = json.loads((a.small_forward / 'report.json').read_text())
            assert small_report['model_sha256'] == sha256(a.small_forward / 'model.pt')
            small_data = np.load(a.small_forward / 'predictions.npz')
            assert np.array_equal(small_data['indices'], align[1]['indices'][pos])
            assert np.array_equal(small_data['labels'], y)
            small = small_data['new_head']
            baseline = pooled_head(controls['base_20261029'], controls['memory_20261029'], small)
            candidate = changed_head(controls['base_20261029'], controls['memory_20261029'], small, new, recipe)
            report['diagnostic'] = {'recipe': recipe, 'baseline_auc': float(roc_auc_score(y, baseline)),
                                    'candidate_auc': float(roc_auc_score(y, candidate)),
                                    'gain': float(roc_auc_score(y, candidate) - roc_auc_score(y, baseline)),
                                    'logloss_change': float(log_loss(y, candidate) - log_loss(y, baseline)),
                                    'scope': 'Thursday-only training, first-Friday head diagnostic; not full V33/V35.'}
            atomic_npz(out / 'predictions.npz', indices=align[1]['indices'][pos], labels=y,
                       baseline=baseline, candidate=candidate, new_head=new,
                       users=np.repeat(align[1]['user'][pos], 10))
        write_json(out / 'report.json', report); print(json.dumps(report), flush=True)
        return
    predictions = {label: np.full(10 * len(position(al)), np.nan) for label, al in zip(labels, align)}
    records = []
    for fold in range(3):
        masks = [al['fold'] != fold for al in align]
        train_users = np.unique(np.concatenate([al['user'][mask] for al, mask in zip(align, masks)]))
        held_users = np.unique(np.concatenate([al['user'][~mask] for al, mask in zip(align, masks)]))
        assert not np.intersect1d(train_users, held_users).size
        path = out / f'fold_{fold}.pt'; completed = out / f'fold_{fold}_completed.json'
        started = time.time()
        if completed.exists():
            record = json.loads(completed.read_text())
            assert record['model_sha256'] == sha256(path)
            model = torch.load(path, map_location='cpu', weights_only=False)
        else:
            if path.exists():
                raise ValueError('Unverified fold checkpoint')
            model = train(ds, masks, **kw)
            temporary = path.with_suffix('.partial'); torch.save(model, temporary); temporary.replace(path)
            record = {'fold': fold, 'training_events': model['train_events'], 'users_disjoint': True,
                      'model_sha256': sha256(path), 'seconds': time.time() - started}
            write_json(completed, record)
        for label, d, al in zip(labels, ds, align):
            pos = position(al); at = np.flatnonzero(al['fold'][pos] == fold)
            rows = (at[:, None] * 10 + np.arange(10)).ravel()
            predictions[label][rows] = predict_events(model, d['x'], pos[at])
        records.append(record); print(record, flush=True)
        del model; gc.collect()
    assert all(np.isfinite(value).all() for value in predictions.values())
    atomic_npz(out / 'predictions.npz', **predictions)
    write_json(out / 'report.json', {'plan': inputs, 'folds': records})


def aggregate(a, plan):
    _, align = load_datasets(a, 388)
    ev = dict(np.load(a.incumbent)); v27 = np.load(a.v27_evidence)
    for key in ['indices', 'labels', 'users', 'period']:
        assert np.array_equal(ev[key], v27[key])
    base = ev['candidate']; parent = v27['baseline']
    old = np.load(a.controls / 'v27_predictions.npz')
    old_memory = np.load(a.controls / 'v29_weekday_predictions.npz')
    memory_weekend = np.load(a.memory_predictions)
    assert np.array_equal(memory_weekend['indices'], ev['indices'])
    memory = {**{name: old_memory[name] for name in ['thursday', 'friday']}, 'weekend': memory_weekend['prediction']}
    small = np.load(a.small_predictions)
    assert np.max(np.abs(base - (.65625 * parent + .109375 * old['weekend'] + .109375 * memory['weekend'] + .125 * small['weekend']))) < 3e-16
    masks = {'combined': np.ones(len(base), bool), 'earlier': ev['period'] == 0, 'later': ev['period'] == 1}
    baseline = {name: float(roc_auc_score(ev['labels'][mask], base[mask])) for name, mask in masks.items()}
    results = []; predictions = {}
    for variant in plan['variants']:
        prediction = np.load(a.out / variant / 'predictions.npz')
        for recipe in plan['mixtures']:
            mode, weight = recipe['mode'], recipe['weight']
            candidate = full_prediction(parent, old['weekend'], memory['weekend'], small['weekend'], prediction['weekend'], recipe)
            assert np.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1
            assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
            gains = {name: float(roc_auc_score(ev['labels'][mask], candidate[mask]) - baseline[name]) for name, mask in masks.items()}
            weekdays = {}
            for k, label in enumerate(['thursday', 'friday']):
                pos = position(align[k]); rows = (pos[:, None] * 10 + np.arange(10)).ravel()
                y = align[k]['labels'][rows]; old_half = pooled_head(old[label], memory[label], small[label])
                new_half = changed_head(old[label], memory[label], small[label], prediction[label], recipe)
                weekdays[label] = float(roc_auc_score(y, new_half) - roc_auc_score(y, old_half))
            loss = float(log_loss(ev['labels'], candidate) - log_loss(ev['labels'], base))
            name = f'{variant}_{mode}_{weight}'
            eligible = all(gains[k] > 0 for k in ['earlier', 'later']) and gains['combined'] >= .00001 and loss <= 0 and all(x > 0 for x in weekdays.values())
            results.append({'name': name, 'variant': variant, 'mode': mode, 'weight': weight, 'gains': gains,
                            'weekday_head_gains': weekdays, 'logloss_change': loss, 'eligible_before_confirmation': eligible})
            predictions[name] = candidate
    eligible = [row for row in results if row['eligible_before_confirmation']]
    selected = None
    if eligible:
        best = max(row['gains']['combined'] for row in eligible)
        near = [row for row in eligible if row['gains']['combined'] >= best - .000005]
        priority = {'replace_small_head': 0, 'replace_pooled_heads': 1, 'blend': 2}
        selected = min(near, key=lambda row: (priority[row['mode']], -row['gains']['combined']))
        atomic_npz(a.out / 'selected_evidence.npz', indices=ev['indices'], labels=ev['labels'], baseline=base,
                   candidate=predictions[selected['name']], users=ev['users'], period=ev['period'])
    report = {'plan': plan, 'incumbent': 'v33', 'official_incumbent_auc': .832616, 'baseline_auc': baseline,
              'results': results, 'selected': selected, 'portal_uploads': 0,
              'limits': 'Matched full-ensemble comparison; weekdays compare the exact normalized pooled-head share only, omitting the V20 parent. Reused development cohorts.'}
    write_json(a.out / 'report.json', report); print(json.dumps(report), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['weekdays', 'weekends', 'incumbent', 'v27-evidence', 'memory-predictions', 'small-predictions', 'small-forward', 'controls', 'plan', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--variant'); p.add_argument('--aggregate', action='store_true')
    p.add_argument('--full', action='store_true'); p.add_argument('--forward', action='store_true')
    p.add_argument('--forward-controls', type=Path)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    if a.full and a.forward:
        raise ValueError('Choose full or forward mode')
    plan = json.loads(a.plan.read_text())
    if a.aggregate:
        aggregate(a, plan)
    else:
        fit(a, plan)


if __name__ == '__main__':
    main()
