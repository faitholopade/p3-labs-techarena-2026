"""Fixed event-distribution pooling alternatives for V45's changed components."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from rebuild_benchmark import sha, write
from research_v43 import WEIGHTS, MASS
from research_v45 import mix as v45_mix


def pool(parts, kind):
    weights = np.asarray(list(WEIGHTS.values())) / MASS
    p = np.stack([parts[k].reshape(-1, 10) for k in WEIGHTS])
    mass = p.sum(2, keepdims=True)
    assert np.isfinite(p).all() and p.min() >= 0 and mass.max() <= 1 + 1e-8
    average = np.einsum('k,kni->ni', weights, p)
    clip = lambda v: np.clip(v, 1e-12, 1 - 1e-12)
    if kind == 'joint':
        simplex = np.concatenate([p, np.maximum(1 - mass, 0)], axis=2)
        logs = np.einsum('k,kni->ni', weights, np.log(clip(simplex)))
        logs -= logs.max(1, keepdims=True); values = np.exp(logs)
        return (values[:, :10] / values.sum(1, keepdims=True)).ravel()
    if kind == 'conditional':
        logs = np.einsum('k,kni->ni', weights, np.log(clip(p / np.maximum(mass, 1e-12))))
        logs -= logs.max(1, keepdims=True); values = np.exp(logs)
        return (values / values.sum(1, keepdims=True) * average.sum(1, keepdims=True)).ravel()
    assert kind == 'event'
    logits = np.log(clip(mass)) - np.log(clip(1 - mass))
    logits = np.einsum('k,kni->ni', weights, logits)
    new_mass = 1 / (1 + np.exp(-logits))
    return (average / np.maximum(average.sum(1, keepdims=True), 1e-12) * new_mass).ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v48_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    evidence = dict(np.load(args.v45 / 'selected_evidence.npz')); y = evidence['labels']; baseline = evidence['candidate']
    parts = {day: dict(np.load(args.benchmark / f'predictions/{day}.npz')) for day in ['weekend', 'thursday', 'friday']}
    heads = {name: dict(np.load(args.v45 / name / 'folds/predictions.npz')) for name in ['plain', 'slate']}
    bases = {'weekend': baseline}
    for day in parts:
        if day != 'weekend':
            ref = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            bases[day] = v45_mix(ref['candidate'], parts[day], {k: v[day] for k, v in heads.items()}, {'mode': 'replace_both'}, MASS)
        parts[day]['small'] = heads['plain'][day]; parts[day]['slate'] = heads['slate'][day]
    arithmetic = {day: sum(w * part[k] for k, w in WEIGHTS.items()) / MASS for day, part in parts.items()}
    masks = {'combined': np.ones(len(y), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        delta = {day: recipe['fraction'] * (pool(part, recipe['kind']) - arithmetic[day]) for day, part in parts.items()}
        candidate = baseline + MASS * delta['weekend']
        assert candidate.min() >= 0 and candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(y[m], candidate[m]) - roc_auc_score(y[m], baseline[m])) for name, m in masks.items()}
        loss = float(log_loss(y, candidate) - log_loss(y, baseline)); weekdays = {}
        for day in ['thursday', 'friday']:
            ref = np.load(args.benchmark / f'inputs/evidence/{day}_components.npz')
            weekdays[day] = float(roc_auc_score(ref['labels'], bases[day] + delta[day]) - roc_auc_score(ref['labels'], bases[day]))
        eligible = gains['combined'] >= .000010 and min(gains.values()) > 0 and loss <= 0 and min(weekdays.values()) > 0
        rows.append(dict(recipe=recipe, gains=gains, logloss_change=loss, weekday_component_gains=weekdays, eligible=eligible)); candidates[recipe['name']] = candidate
    valid = [r for r in rows if r['eligible']]; selected = None
    if valid:
        best = max(r['gains']['combined'] for r in valid)
        selected = next(r for r in rows if r['eligible'] and r['gains']['combined'] >= best - .000005)
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=y, users=evidence['users'], period=evidence['period'], baseline=baseline, candidate=candidates[selected['recipe']['name']])
    report = dict(plan=plan, plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)), baseline_auc=float(roc_auc_score(y, baseline)), results=rows, selected=selected, official_score=None, portal_uploads=0)
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
