"""Rebuild the scored V45 development baseline from shared saved heads, without retraining."""
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
from research_v43 import positions
from research_v45 import mix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((REPO / 'artifacts/v45/research/index.json').read_text())
    expected = {row['path']: row['sha256'] for row in index['models']}
    frozen = json.loads((REPO / 'artifacts/v45/research/plain/folds/inputs.json').read_text())
    arrays, alignment, hashes, feature_hashes = {}, {}, {}, {}
    for day in ['thursday', 'friday', 'weekend']:
        folder = args.benchmark / 'features' / day
        feature_hashes[day] = {name: sha(folder / name) for name in ['raw.npy', 'alignment.npz', 'names.json']}
        assert feature_hashes[day] == frozen['features'][day], day
        arrays[day] = np.load(folder / 'raw.npy', mmap_mode='r')
        alignment[day] = dict(np.load(folder / 'alignment.npz'))
    all_predictions = {}
    for family in ['plain', 'slate']:
        predictions = {day: np.full(len(positions(a)) * 10, np.nan) for day, a in alignment.items()}
        for fold in range(3):
            rel = f'artifacts/v45/research/{family}/folds/fold_{fold}.pt'
            path = REPO / rel
            hashes[rel] = sha(path); assert hashes[rel] == expected[rel]
            model = torch.load(path, map_location='cpu', weights_only=False)
            for day, raw in arrays.items():
                a = alignment[day]; pos = positions(a); test = np.flatnonzero(a['fold'][pos] == fold)
                for start in range(0, len(test), 1500):
                    events = test[start:start + 1500]
                    rows = (pos[events, None] * 10 + np.arange(10)).ravel()
                    dest = (events[:, None] * 10 + np.arange(10)).ravel()
                    predictions[day][dest] = predict(model, raw[rows])
        assert all(np.isfinite(value).all() for value in predictions.values())
        folder = args.out / family / 'folds'; folder.mkdir(parents=True, exist_ok=True)
        destination = folder / 'predictions.npz'
        if destination.exists():
            previous = np.load(destination)
            assert all(np.array_equal(previous[k], v) for k, v in predictions.items())
        np.savez_compressed(destination, **predictions)
        all_predictions[family] = predictions
        rel = f'artifacts/v45/research/{family}/forward/model.pt'
        hashes[rel] = sha(REPO / rel); assert hashes[rel] == expected[rel]
        model = torch.load(REPO / rel, map_location='cpu', weights_only=False)
        a = alignment['friday']; pos = positions(a); parts = []
        for start in range(0, len(pos), 1500):
            rows = (pos[start:start + 1500, None] * 10 + np.arange(10)).ravel()
            parts.append(predict(model, arrays['friday'][rows]))
        folder = args.out / family / 'forward'; folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / 'predictions.npz', prediction=np.concatenate(parts), indices=a['indices'][pos])
    reference = dict(np.load(args.benchmark / 'inputs/evidence/matched_v38.npz'))
    controls = dict(np.load(args.benchmark / 'predictions/weekend.npz'))
    candidate = mix(reference['candidate'], controls,
                    {k: v['weekend'] for k, v in all_predictions.items()}, {'mode': 'replace_both'})
    auc = float(roc_auc_score(reference['labels'], candidate))
    assert abs(auc - .8378046551733912) < 1e-12
    path = args.out / 'selected_evidence.npz'
    if path.exists():
        assert np.array_equal(np.load(path)['candidate'], candidate)
    np.savez_compressed(path, indices=reference['indices'], labels=reference['labels'], users=reference['users'],
                        period=reference['period'], baseline=reference['candidate'], candidate=candidate)
    report = dict(incumbent='v45', official_auc=.832655, matched_auc=auc, model_sha256=hashes,
                  feature_sha256=feature_hashes,
                  source_sha256=sha(Path(__file__)), scope='Saved held-user predictions and exact fixed V45 recipe; no fitting.')
    write(args.out / 'baseline_reproduction.json', report); print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
