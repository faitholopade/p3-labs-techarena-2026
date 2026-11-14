"""Reproduce V12 from the V11 runtime and the compact V12 model extension."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infer_v8 import predict as predict_v8
from infer_v11 import factor_predictions
from daily_inference import build
from factor_ensemble import update_features
from calibration_model import predict as predict_head
from validate_submission import validate


def components(root, extension, threads=3):
    root, extension = Path(root), Path(extension)
    prior_spec = json.loads((root / 'models/v11_spec.json').read_text())
    spec = json.loads((extension / 'models/v12_spec.json').read_text())
    factor = extension / 'models' / spec['rank_factor_model']
    with factor.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != spec['rank_factor_sha256']:
        raise ValueError('V12 ranking checkpoint SHA-256 mismatch.')
    if prior_spec != spec['required_v11_spec']:
        raise ValueError('The V11 ensemble specification differs from the required baseline.')
    ids, baseline = predict_v8(root, threads)
    gc.collect()
    train = dict(np.load(root / 'data/train.npz'))
    events = dict(np.load(root / 'data/evaluation.npz'))
    audit = json.loads((root / 'data/audit.json').read_text())
    features, names = build(train, events, 1737936000, audit['n_items'], audit['n_users'],
                            root / 'models' / prior_spec['factor_models'][0],
                            feature_state=root / 'models/feature_state.npz',
                            exposure_basis=root / 'data/exposure_basis.npz')
    original_head = factor_predictions(root, features, events, names)
    update_features(features, events, factor)
    heads = [torch.load(root / 'models' / name, map_location='cpu', weights_only=False)
             for name in prior_spec['calibration_models']]
    for head in heads:
        if head['names'] != names:
            raise ValueError('Feature names differ from the saved calibration schema.')
    rank_head = np.mean([predict_head(head, features) for head in heads], axis=0)
    weights = np.asarray(spec['weights'], dtype=float)
    if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Expected three nonnegative weights summing to one.')
    probability = weights[0]*baseline + weights[1]*original_head + weights[2]*rank_head
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert probability.reshape(-1, 10).sum(axis=1).max() <= 1 + 1e-12
    return ids, probability, baseline, features, events, names


def predict(root, extension, threads=3):
    values = components(root, extension, threads)
    return values[0], values[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Extracted V11 runtime.')
    parser.add_argument('--extension', required=True, help='Extracted V12 model extension.')
    parser.add_argument('--out', default='P3_Labs_submission_v12.csv')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    ids, probability = predict(args.root, args.extension, args.threads)
    pd.DataFrame({'rowid': ids, 'predicted_probability': probability}).to_csv(
        args.out, index=False, float_format='%.10g')
    report = validate(args.out, ids)
    assert report['reference_order_matches']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()

