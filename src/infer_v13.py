"""Reproduce V13 from the V11 runtime plus the V12 and V13 model extensions."""
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
from sequence_calibration import history_pairs, predict as predict_sequence
from validate_submission import validate


def verify_hash(path, expected):
    with Path(path).open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != expected:
        raise ValueError(f'Model SHA-256 mismatch: {Path(path).name}')


def predict(root, v12_extension, extension, threads=3):
    root, v12_extension, extension = Path(root), Path(v12_extension), Path(extension)
    prior = json.loads((root / 'models/v11_spec.json').read_text())
    v12 = json.loads((v12_extension / 'models/v12_spec.json').read_text())
    spec = json.loads((extension / 'models/v13_spec.json').read_text())
    if prior != v12['required_v11_spec'] or v12 != spec['required_v12_spec']:
        raise ValueError('The baseline ensemble specifications differ from the required versions.')
    rank_factor = v12_extension / 'models' / v12['rank_factor_model']
    sequence_factor = root / 'models' / spec['sequence_factor_model']
    verify_hash(rank_factor, v12['rank_factor_sha256'])
    verify_hash(sequence_factor, spec['sequence_factor_sha256'])
    sequence_heads = []
    for filename in spec['sequence_models']:
        path = extension / 'models' / filename
        verify_hash(path, spec['sequence_model_sha256'][filename])
        sequence_heads.append(torch.load(path, map_location='cpu', weights_only=False))
    ids, legacy = predict_v8(root, threads)
    gc.collect()
    train = dict(np.load(root / 'data/train.npz'))
    events = dict(np.load(root / 'data/evaluation.npz'))
    audit = json.loads((root / 'data/audit.json').read_text())
    raw, names = build(train, events, 1737936000, audit['n_items'], audit['n_users'],
                       root / 'models' / prior['factor_models'][0],
                       feature_state=root / 'models/feature_state.npz',
                       exposure_basis=root / 'data/exposure_basis.npz')
    original = factor_predictions(root, raw, events, names)
    heads = [torch.load(root / 'models' / name, map_location='cpu', weights_only=False)
             for name in prior['calibration_models']]
    if any(head['names'] != names for head in heads):
        raise ValueError('Factor-head feature schema mismatch.')
    if any(head['base_spec']['names'] != names for head in sequence_heads):
        raise ValueError('Sequence-head feature schema mismatch.')
    update_features(raw, events, rank_factor)
    rank = np.mean([predict_head(head, raw) for head in heads], axis=0)
    w12 = np.asarray(v12['weights'], dtype=float)
    baseline = w12[0]*legacy + w12[1]*original + w12[2]*rank
    del original, rank, train
    gc.collect()
    update_features(raw, events, sequence_factor)
    state = torch.load(sequence_factor, map_location='cpu', weights_only=False)['state']
    pairs, mask = history_pairs(state, events)
    sequence = np.mean([predict_sequence(head, raw, pairs, mask) for head in sequence_heads], axis=0)
    weights = np.asarray(spec['weights'], dtype=float)
    if weights.shape != (3,) or (weights < 0).any() or not np.isclose(weights.sum(), 1):
        raise ValueError('Expected three nonnegative weights summing to one.')
    probability = weights[0]*baseline + weights[1]*legacy + weights[2]*sequence
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert probability.reshape(-1, 10).sum(axis=1).max() <= 1 + 1e-12
    return ids, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Extracted V11 runtime.')
    parser.add_argument('--v12-extension', required=True)
    parser.add_argument('--extension', required=True, help='Extracted V13 model extension.')
    parser.add_argument('--out', default='P3_Labs_submission_v13.csv')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    ids, probability = predict(args.root, args.v12_extension, args.extension, args.threads)
    pd.DataFrame({'rowid': ids, 'predicted_probability': probability}).to_csv(
        args.out, index=False, float_format='%.10g')
    report = validate(args.out, ids)
    assert report['reference_order_matches']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
