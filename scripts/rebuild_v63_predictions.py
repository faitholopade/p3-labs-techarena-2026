"""Reconstruct the scored V63 development baseline from shared saved models."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from calibration_model import predict
from rebuild_benchmark import sha, write
from research_v43 import positions, MASS
from research_v45 import mix as v45_mix
from research_v51 import correction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v51', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    index = json.loads((REPO / 'artifacts/v63/research/index.json').read_text())
    expected = {row['path']: row['sha256'] for row in index['models']}
    frozen = json.loads((REPO / 'artifacts/v63/research/folds/fold_0_inputs.json').read_text())
    recipe = json.loads((REPO / 'artifacts/v63/spec.json').read_text())['recipe']
    raw, align, hashes = {}, {}, {}
    for day in ['thursday', 'friday', 'weekend']:
        folder = args.benchmark / 'features' / day
        hashes[day] = {name: sha(folder / name) for name in ['raw.npy', 'alignment.npz', 'names.json']}
        assert hashes[day] == frozen['features'][day]
        raw[day] = np.load(folder / 'raw.npy', mmap_mode='r')
        align[day] = dict(np.load(folder / 'alignment.npz'))
    predictions = {day: np.full(len(positions(a)) * 10, np.nan) for day, a in align.items()}
    model_hashes = {}
    for fold in range(3):
        rel = f'artifacts/v63/research/folds/fold_{fold}.pt'
        model_hashes[rel] = sha(REPO / rel); assert model_hashes[rel] == expected[rel]
        model = torch.load(REPO / rel, map_location='cpu', weights_only=False)
        for day, a in align.items():
            pos = positions(a); test = np.flatnonzero(a['fold'][pos] == fold)
            for start in range(0, len(test), 1500):
                events = test[start:start + 1500]
                rows = (pos[events, None] * 10 + np.arange(10)).ravel()
                dest = (events[:, None] * 10 + np.arange(10)).ravel()
                predictions[day][dest] = predict(model, raw[day][rows])
    assert all(np.isfinite(v).all() for v in predictions.values())
    np.savez_compressed(args.out / 'adapted_predictions.npz', **predictions)
    evidence = dict(np.load(args.v51 / 'selected_evidence.npz'))
    previous = dict(np.load(args.v51 / 'current_predictions.npz'))
    current = {}
    for day in predictions:
        controls = dict(np.load(args.benchmark / f'predictions/{day}.npz'))
        change = .0478515625 * (predictions[day]-controls['base']) + .0478515625 * (predictions[day]-controls['memory'])
        current[day] = previous[day] + change / (1. if day=='weekend' else MASS)
    auc = float(roc_auc_score(evidence['labels'], current['weekend']))
    assert abs(auc - .8378351482608294) < 1e-12
    path = args.out / 'selected_evidence.npz'
    if path.exists(): assert np.array_equal(np.load(path)['candidate'], current['weekend'])
    np.savez_compressed(path, indices=evidence['indices'], labels=evidence['labels'], users=evidence['users'],
                        period=evidence['period'], baseline=evidence['candidate'], candidate=current['weekend'])
    np.savez_compressed(args.out / 'current_predictions.npz', **current)
    report = dict(incumbent='v63', official_auc=.832698, matched_auc=auc,
                  model_sha256=model_hashes, feature_sha256=hashes, source_sha256=sha(Path(__file__)),
                  scope='Weekend is full scored V63; Thursday/Friday are normalized changed-component diagnostics. Saved heads only; no fitting.')
    write(args.out / 'baseline_reproduction.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
