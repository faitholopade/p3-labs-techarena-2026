"""Reproduce V10 using three factor fits and the frozen V9 calibration heads."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infer_v8 import predict as predict_v8
from daily_inference import build
from factor_ensemble import update_features
from calibration_model import predict as predict_head
from validate_submission import validate


def factor_predictions(root, features, events, names):
    root = Path(root)
    spec = json.loads((root / 'models/v10_spec.json').read_text())
    heads = [torch.load(root / 'models' / name, map_location='cpu', weights_only=False)
             for name in spec['calibration_models']]
    for head in heads:
        if head['names'] != names:
            raise ValueError('Feature names differ from the saved calibration schema.')
    predictions = []
    for index, name in enumerate(spec['factor_models']):
        if index:
            update_features(features, events, root / 'models' / name)
        predictions.append(np.mean([predict_head(head, features) for head in heads], axis=0))
    return np.average(predictions, axis=0, weights=spec['factor_weights'])


def predict(root, threads=3):
    root = Path(root)
    ids, baseline = predict_v8(root, threads)
    gc.collect()
    train = dict(np.load(root / 'data/train.npz'))
    events = dict(np.load(root / 'data/evaluation.npz'))
    audit = json.loads((root / 'data/audit.json').read_text())
    spec = json.loads((root / 'models/v10_spec.json').read_text())
    features, names = build(train, events, 1737936000, audit['n_items'], audit['n_users'],
                            root / 'models' / spec['factor_models'][0],
                            feature_state=root / 'models/feature_state.npz',
                            exposure_basis=root / 'data/exposure_basis.npz')
    head = factor_predictions(root, features, events, names)
    weight = spec['new_head_weight']
    probability = (1 - weight) * baseline + weight * head
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert probability.reshape(-1, 10).sum(axis=1).max() <= 1 + 1e-12
    return ids, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='.')
    parser.add_argument('--out', default='P3_Labs_submission_v10.csv')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    ids, probability = predict(args.root, args.threads)
    pd.DataFrame({'rowid': ids, 'predicted_probability': probability}).to_csv(
        args.out, index=False, float_format='%.10g')
    report = validate(args.out, ids)
    assert report['reference_order_matches']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
