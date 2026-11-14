"""Fit the unchanged full V47 tree with an explicit runtime thread budget."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
from research_v47 import fit
from rebuild_benchmark import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v45', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    args.stage = 'full'; args.plan = REPO / 'results/v47_experiment_plan.json'
    plan = json.loads(args.plan.read_text())
    assert args.threads >= 1
    plan['training']['threads'] = args.threads
    write(args.out / 'full_runtime.json', {
        'statistical_plan_sha256': sha(args.plan), 'runtime_threads': args.threads,
        'source_sha256': sha(Path(__file__)), 'stage': 'full',
        'reason': 'Explicit CPU allocation. Deterministic LightGBM; seed, bins, feature sampling, objective and all statistical hyperparameters unchanged.',
        'timing': 'Prefitted before V47 screen scores to reduce release latency; this is not a release decision.'})
    fit(args, plan)


if __name__ == '__main__': main()
