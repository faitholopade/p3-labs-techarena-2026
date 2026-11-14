"""Reproduce the fixed, single-weekday paired user-bootstrap assessment."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
from assess_candidate import prepared_auc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--replicates', type=int, default=400)
    parser.add_argument('--seed', type=int, default=20261029)
    args = parser.parse_args()
    if args.replicates < 1:
        raise ValueError('Replicate count must be positive.')
    z = np.load(args.evidence)
    y, baseline, candidate = z['labels'], z['baseline'], z['candidate']
    users = z['users'] if 'users' in z else z['user']
    if not all(x.shape == y.shape for x in [baseline, candidate, users]):
        raise ValueError('Evidence arrays do not align.')
    if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
        raise ValueError('Invalid candidate probabilities.')
    _, groups = np.unique(users, return_inverse=True)
    old, new = prepared_auc(y, baseline), prepared_auc(y, candidate)
    ones = np.ones(len(y))
    assert abs(old(ones) - roc_auc_score(y, baseline)) < 1e-12
    assert abs(new(ones) - roc_auc_score(y, candidate)) < 1e-12
    rng = np.random.default_rng(args.seed)
    deltas = []
    for _ in range(args.replicates):
        weights = rng.poisson(1, groups.max() + 1)[groups].astype(float)
        deltas.append(new(weights) - old(weights))
    report = {
        'auc_improvement': new(ones) - old(ones),
        'interval_95_percent': np.quantile(deltas, [.025, .975]).tolist(),
        'replicates': args.replicates,
        'positive_replicates': int(np.sum(np.array(deltas) > 0)),
        'users': int(groups.max() + 1), 'seed': args.seed,
        'method': 'Paired Poisson bootstrap clustered by user on fresh Friday chronological backtest.',
        'limitation': 'Conditional on fixed learned model and this cohort, not a V20 or public-score prediction; training and distribution-shift uncertainty are not included.'}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
