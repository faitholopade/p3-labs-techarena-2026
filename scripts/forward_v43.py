"""Fixed V43 first-Friday diagnostic using Thursday-only calibration fits."""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import lightgbm as lgb
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from calibration_model import predict
from v30_tree import predict as tree_predict
from rebuild_benchmark import sha, write
from research_v43 import WEIGHTS, MASS, mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'checkpoints', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads((REPO / 'results/v38_forward_diagnostic.json').read_text())
    selected = json.loads((args.checkpoints / 'v43/report.json').read_text())['selected']
    assert selected['recipe']['name'] == 'replace_both'
    folder = args.benchmark / 'features/friday'
    raw = np.load(folder / 'raw.npy', mmap_mode='r')
    alignment = dict(np.load(folder / 'alignment.npz'))
    positions = alignment['first_position']
    rows = (positions[:, None] * 10 + np.arange(10)).ravel()
    labels = alignment['labels'][rows]
    control_hashes = {'base': reference['existing_control_hashes']['base_20261029'],
                      'memory': reference['existing_control_hashes']['memory_20261029'],
                      'small': reference['small_control_sha256']}
    paths = {name: args.checkpoints / 'controls' / (name + '.pt') for name in control_hashes}
    paths['slate'] = REPO / 'artifacts/v38/diagnostics/slate_head.pt'
    index = json.loads((REPO / 'artifacts/v38/diagnostics/index.json').read_text())
    control_hashes['slate'] = next(entry['sha256'] for entry in index['models'] if entry['path'].endswith('slate_head.pt'))
    for name, path in paths.items():
        assert sha(path) == control_hashes[name], name
    for version, kind in [('v42', 'small'), ('v43', 'slate')]:
        for seed in [20261030, 20261031]:
            folder = args.checkpoints / version / f'seed_{seed}/forward'
            completed = json.loads((folder / 'model_completed.json').read_text())
            path = folder / 'model.pt'
            assert sha(path) == completed['sha256'] and completed['training_events'] == 97717
            paths[f'{kind}_{seed}'] = path
    parts = {}
    for name, path in paths.items():
        model = torch.load(path, map_location='cpu', weights_only=False)
        assert model['train_events'] == 97717
        columns = 344 if name == 'base' else 388
        predictions = []
        for start in range(0, len(positions), 2000):
            selected_rows = (positions[start:start + 2000, None] * 10 + np.arange(10)).ravel()
            predictions.append(predict(model, raw[selected_rows, :columns]))
        parts[name] = np.concatenate(predictions)
    tree_path = REPO / 'artifacts/v38/diagnostics/stable_tree.txt'
    expected = next(entry['sha256'] for entry in index['models'] if entry['path'].endswith('stable_tree.txt'))
    assert sha(tree_path) == expected
    tree = lgb.Booster(model_file=str(tree_path))
    tree_spec = json.loads((REPO / 'artifacts/v38/diagnostics/stable_tree_spec.json').read_text())
    parts['tree'] = np.concatenate([tree_predict(tree, raw[rows[start:start + 20000], :344], tree_spec)
                                    for start in range(0, len(rows), 20000)])
    baseline = sum(weight * parts[name] for name, weight in WEIGHTS.items()) / MASS
    baseline_auc = float(roc_auc_score(labels, baseline))
    assert abs(baseline_auc - reference['candidate_auc']) < 1e-8
    parts['small_mean'] = (parts['small'] + parts['small_20261030'] + parts['small_20261031']) / 3
    slate_mean = (parts['slate'] + parts['slate_20261030'] + parts['slate_20261031']) / 3
    candidate = mix(baseline, parts, slate_mean, selected['recipe'], MASS)
    assert np.isfinite(candidate).all() and candidate.min() >= 0
    assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
    candidate_auc = float(roc_auc_score(labels, candidate))
    report = {'recipe': selected['recipe'], 'training_day': 'Thursday', 'evaluation_day': 'Friday',
              'events': len(positions), 'baseline_auc': baseline_auc, 'candidate_auc': candidate_auc,
              'gain': candidate_auc - baseline_auc,
              'logloss_change': float(log_loss(labels, candidate) - log_loss(labels, baseline)),
              'baseline_reproduction_auc_error': abs(baseline_auc - reference['candidate_auc']),
              'model_sha256': {name: sha(path) for name, path in paths.items()},
              'tree_sha256': sha(tree_path), 'feature_sha256': sha(args.benchmark / 'features/friday/raw.npy'),
              'source_sha256': sha(Path(__file__)), 'not_full_incumbent_validation': True,
              'scope': 'Exact normalized changed-component weights; all heads trained only on Thursday. Omits unchanged V17 and V20 neural components. No weights tuned on this result.',
              'official_score': None, 'portal_uploads': 0}
    args.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out / 'evidence.npz', indices=alignment['indices'][positions], labels=labels,
                        users=alignment['user'][positions].repeat(10), baseline=baseline, candidate=candidate)
    write(args.out / 'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
