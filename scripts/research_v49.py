"""Fixed scaling of the two six-day heads' corrections to their original utility offset."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, log_loss
from rebuild_benchmark import sha, write
from research_v43 import WEIGHTS, MASS, positions
from research_v45 import mix as v45_mix


def rescale(prediction, offset, scale):
    p = prediction.reshape(-1, 10)
    outside = 1 - p.sum(1, keepdims=True)
    assert p.min() > 0 and outside.min() > 0
    utility = np.log(p) - np.log(outside)
    z = offset.reshape(-1, 10) + scale * (utility - offset.reshape(-1, 10))
    return softmax(np.c_[z, np.zeros(len(z))], axis=1)[:, :10].ravel()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--plan', type=Path, default=REPO / 'results/v49_experiment_plan.json')
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    evidence = dict(np.load(args.v45 / 'selected_evidence.npz')); y = evidence['labels']; baseline = evidence['candidate']
    heads = {name: dict(np.load(args.v45 / name / 'folds/predictions.npz')) for name in ['plain', 'slate']}
    offsets, weekdays, baselines = {}, {}, {'weekend': baseline}
    hashes, identity_errors = {}, {}
    for day in ['weekend', 'thursday', 'friday']:
        folder = args.benchmark / 'features' / day; a = dict(np.load(folder / 'alignment.npz')); raw = np.load(folder / 'raw.npy', mmap_mode='r')
        names = json.loads((folder / 'names.json').read_text()); col = names.index('posterior1.0_mix1.0_utility')
        pos = positions(a); rows = (pos[:, None] * 10 + np.arange(10)).ravel()
        offsets[day] = np.asarray(raw[rows, col], np.float64)
        hashes[day] = sha(folder / 'alignment.npz')
        for family in heads:
            error = float(np.max(np.abs(rescale(heads[family][day], offsets[day], 1.) - heads[family][day])))
            assert error < 1e-14
            identity_errors[day + '_' + family] = error
        if day != 'weekend':
            weekdays[day] = dict(np.load(args.benchmark / f'inputs/evidence/{day}_components.npz'))
            controls = dict(np.load(args.benchmark / f'predictions/{day}.npz'))
            baselines[day] = v45_mix(weekdays[day]['candidate'], controls, {k: v[day] for k, v in heads.items()}, {'mode': 'replace_both'}, MASS)
    masks = {'combined': np.ones(len(y), bool), 'earlier': evidence['period'] == 0, 'later': evidence['period'] == 1}
    rows, candidates = [], {}
    for recipe in plan['recipes']:
        delta = {day: sum(WEIGHTS[key] * (rescale(heads[family][day], offsets[day], recipe['scale']) - heads[family][day])
                         for key, family in [('small', 'plain'), ('slate', 'slate')]) for day in offsets}
        candidate = baseline + delta['weekend']
        assert candidate.min() >= 0 and candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
        gains = {name: float(roc_auc_score(y[m], candidate[m]) - roc_auc_score(y[m], baseline[m])) for name, m in masks.items()}
        loss = float(log_loss(y, candidate) - log_loss(y, baseline))
        checks = {day: float(roc_auc_score(weekdays[day]['labels'], baselines[day] + delta[day] / MASS) - roc_auc_score(weekdays[day]['labels'], baselines[day])) for day in weekdays}
        eligible = gains['combined'] >= .000010 and min(gains.values()) > 0 and loss <= 0 and min(checks.values()) > 0
        rows.append(dict(recipe=recipe, gains=gains, logloss_change=loss, weekday_component_gains=checks, eligible=eligible)); candidates[recipe['name']] = candidate
    valid = [r for r in rows if r['eligible']]; selected = None
    if valid:
        best = max(r['gains']['combined'] for r in valid)
        selected = min([r for r in valid if r['gains']['combined'] >= best - .000005], key=lambda r: abs(r['recipe']['scale'] - 1))
        np.savez_compressed(args.out / 'selected_evidence.npz', indices=evidence['indices'], labels=y, users=evidence['users'], period=evidence['period'], baseline=baseline, candidate=candidates[selected['recipe']['name']])
    report = dict(plan=plan, plan_sha256=sha(args.plan), source_sha256=sha(Path(__file__)), alignment_sha256=hashes,
                  identity_errors=identity_errors, baseline_auc=float(roc_auc_score(y, baseline)), results=rows, selected=selected, official_score=None, portal_uploads=0)
    write(args.out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
