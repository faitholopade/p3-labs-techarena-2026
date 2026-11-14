"""Run the frozen forward workflow with the lossless, identity-checked storage backend."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import research_v45
from v45_compact_training import train
from rebuild_benchmark import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--family', choices=['plain', 'slate'], required=True)
    args = parser.parse_args()
    report = {'family': args.family, 'backend': 'lossless repeated-event-column storage',
              'identity_report_sha256': sha(REPO / 'results/v45_compact_identity.json'),
              'sources': {name: sha(REPO / name) for name in ['scripts/run_v45_compact_forward.py',
                  'src/v45_compact_training.py', 'src/v45_training.py', 'scripts/research_v45.py']},
              'scope': 'Only feature storage differs; fixed model, seed, objective and folds are unchanged. Tuesday, Wednesday and Thursday only; Friday metrics are assessed after recipe selection.'}
    write(args.out / args.family / 'compact_forward_execution.json', report)
    research_v45.dense_train = train
    research_v45.train = train
    sys.argv = ['research_v45.py', '--benchmark', str(args.benchmark), '--out', str(args.out),
                '--family', args.family, '--stage', 'forward']
    research_v45.main()


if __name__ == '__main__':
    main()
