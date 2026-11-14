"""Fixed event-mass/conditional-choice decompositions against scored V31."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def decompose(base, memory):
    a = np.asarray(base, dtype=np.float64).reshape(-1, 10)
    b = np.asarray(memory, dtype=np.float64).reshape(-1, 10)
    qa, qb = a.sum(1, keepdims=True), b.sum(1, keepdims=True)
    assert np.all(qa > 0) and np.all(qb > 0)
    average = .5 * (a + b)
    q = average.sum(1, keepdims=True)
    ra, rb = a / qa, b / qb
    geometric = np.sqrt(ra * rb)
    geometric /= geometric.sum(1, keepdims=True)
    return {
        'arithmetic_conditional_choice': (q * .5 * (ra + rb)).ravel(),
        'geometric_conditional_choice': (q * geometric).ravel(),
        'base_event_rate': (qa * average / q).ravel(),
        'memory_event_rate': (qb * average / q).ravel(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['plan', 'v31', 'v27', 'v29', 'controls', 'weekdays', 'forward', 'out']:
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(a.plan.read_text())
    ev = dict(np.load(a.v31)); old = np.load(a.v27); memory = np.load(a.v29)
    assert np.array_equal(old['indices'], ev['indices'])
    assert np.array_equal(memory['indices'], ev['indices'])
    # V29 evidence baseline is V27; its new head is recoverable without labels.
    parent = old['baseline']
    hbase = (old['candidate'] - .75 * parent) / .25
    hmemory = (memory['candidate'] - .75 * parent) / .25
    assert np.max(np.abs(.75 * parent + .125 * hbase + .125 * hmemory - ev['candidate'])) < 3e-16
    candidates = {name: .75 * parent + .25 * head for name, head in decompose(hbase, hmemory).items()}
    controls = np.load(a.controls / 'v27_predictions.npz')
    memories = np.load(a.controls / 'v29_weekday_predictions.npz')
    weekday = {}
    for name in ['thursday', 'friday']:
        alignment = np.load(a.weekdays / name / 'alignment.npz')
        rows = (alignment['first_position'][:, None] * 10 + np.arange(10)).ravel()
        labels = alignment['labels'][rows]
        baseline = .5 * controls[name] + .5 * memories[name]
        auc = roc_auc_score(labels, baseline)
        weekday[name] = {variant: float(roc_auc_score(labels, prediction) - auc)
                         for variant, prediction in decompose(controls[name], memories[name]).items()}
    forward = np.load(a.forward)
    fbase = forward['base_20261029']; fmemory = forward['memory_20261029']
    al = np.load(a.weekdays / 'friday/alignment.npz')
    rows = (al['first_position'][:, None] * 10 + np.arange(10)).ravel()
    fy = al['labels'][rows]
    fhead = .5 * fbase + .5 * fmemory
    faux = roc_auc_score(fy, fhead)
    forward_gains = {variant: float(roc_auc_score(fy, prediction) - faux)
                     for variant, prediction in decompose(fbase, fmemory).items()}
    masks = {'combined': np.ones(len(ev['labels']), bool), 'earlier': ev['period'] == 0, 'later': ev['period'] == 1}
    baseline = {name: float(roc_auc_score(ev['labels'][mask], ev['candidate'][mask])) for name, mask in masks.items()}
    results = []
    for variant in plan['variants']:
        prediction = candidates[variant]
        assert np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1
        assert prediction.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(ev['labels'][mask], prediction[mask]) - baseline[name]) for name, mask in masks.items()}
        change = float(log_loss(ev['labels'], prediction) - log_loss(ev['labels'], ev['candidate']))
        weekdays = {name: weekday[name][variant] for name in weekday}
        passes = all(gains[name] > 0 for name in ['earlier', 'later']) and gains['combined'] >= .00001 and change <= 0 and all(x > 0 for x in weekdays.values()) and forward_gains[variant] > 0
        results.append({'variant': variant, 'matched_gains': gains, 'logloss_change': change, 'weekday_head_gains': weekdays, 'forward_head_gain': forward_gains[variant], 'eligible_before_bootstrap': passes})
    eligible = [row for row in results if row['eligible_before_bootstrap']]
    selected = max(eligible, key=lambda row: row['matched_gains']['combined']) if eligible else None
    if selected:
        np.savez_compressed(a.out / 'evidence.npz', **{**ev, 'baseline': ev['candidate'], 'candidate': candidates[selected['variant']]})
    record = {'plan': plan, 'plan_sha256': sha(a.plan), 'source_sha256': sha(Path(__file__)),
              'inputs_sha256': {name: sha(getattr(a, name)) for name in ['v31', 'v27', 'v29', 'forward']},
              'baseline_auc': baseline, 'results': results, 'selected': selected, 'portal_uploads': 0}
    (a.out / 'report.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
