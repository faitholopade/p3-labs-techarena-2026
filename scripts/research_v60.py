"""Fixed audit of the oldest ensemble contribution, without new fitting."""
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'scripts')]
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from rebuild_benchmark import sha, write


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['benchmark', 'v45', 'v51', 'out']:
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    path = REPO / 'results/v60_experiment_plan.json'; plan = json.loads(path.read_text())
    ev = dict(np.load(a.v51 / 'selected_evidence.npz')); base = ev['candidate']; y = ev['labels']
    original = dict(np.load(a.benchmark / 'predictions/weekend.npz'))
    # Algebraically recover the unchanged V17 component from the verified V38 blend.
    weights = dict(base=.095703125, memory=.095703125, small=.109375,
                   slate=.125, tree=.06280517578125, neural32=.07177734375)
    legacy = (original['candidate'] - sum(w * original[k] for k, w in weights.items())) / plan['parent_weight']
    assert min(legacy) >= 0 and legacy.reshape(-1, 10).sum(1).max() <= 1 + 1e-10
    assert np.max(abs(plan['parent_weight'] * legacy + sum(w * original[k] for k, w in weights.items()) - original['candidate'])) < 5e-16
    narrow = np.load(a.v51 / 'narrow_predictions.npz')['weekend']
    plain = np.load(a.v45 / 'plain/folds/predictions.npz')['weekend']
    slate = np.load(a.v45 / 'slate/folds/predictions.npz')['weekend']
    targets = {'narrow': narrow, 'three_head_mean': (narrow + plain + slate) / 3}
    masks = {'combined': np.ones(len(y), bool), 'earlier': ev['period'] == 0, 'later': ev['period'] == 1}
    rows = []; candidates = {}
    for recipe in plan['recipes']:
        candidate = base + plan['parent_weight'] * recipe['fraction'] * (targets[recipe['target']] - legacy)
        gains = {k: float(roc_auc_score(y[m], candidate[m]) - roc_auc_score(y[m], base[m])) for k, m in masks.items()}
        loss = float(log_loss(y, candidate) - log_loss(y, base))
        eligible = gains['combined'] >= .000010 and min(gains.values()) > 0 and loss <= 0
        rows.append(dict(recipe=recipe, gains=gains, logloss_change=loss, eligible=eligible))
        candidates[recipe['name']] = candidate
    valid = [r for r in rows if r['eligible']]; selected = None
    if valid:
        best = max(r['gains']['combined'] for r in valid)
        selected = min([r for r in valid if r['gains']['combined'] >= best - .000005], key=lambda r: r['recipe']['fraction'])
        np.savez_compressed(a.out / 'selected_evidence.npz', **{k: ev[k] for k in ['indices', 'users', 'labels', 'period']}, baseline=base, candidate=candidates[selected['recipe']['name']])
    report = dict(plan=plan, plan_sha256=sha(path), source_sha256=sha(Path(__file__)),
                  baseline_auc=float(roc_auc_score(y, base)), results=rows, selected=selected,
                  weekday_and_forward_scores=None, official_score=None, portal_uploads=0)
    write(a.out / 'report.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
