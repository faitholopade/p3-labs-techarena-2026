"""Reproduce V14 with higher-precision posterior estimates and rate calibration."""
import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infer_v8 import predict as predict_v8
from infer_v13 import verify_hash
from daily_inference import build
from factor_ensemble import update_features
from posterior_inference import features as posterior_features
from calibration_model import predict as predict_head
from sequence_calibration import history_pairs, predict as predict_sequence
from rate_calibration import predict as predict_rate
from rate_features import features as rate_features
from validate_submission import validate


def predict(root, v12_extension, v13_extension, extension, threads=3):
    root, v12_extension = Path(root), Path(v12_extension)
    v13_extension, extension = Path(v13_extension), Path(extension)
    v11 = json.loads((root / 'models/v11_spec.json').read_text())
    v12 = json.loads((v12_extension / 'models/v12_spec.json').read_text())
    v13 = json.loads((v13_extension / 'models/v13_spec.json').read_text())
    spec = json.loads((extension / 'models/v14_spec.json').read_text())
    if v11 != v12['required_v11_spec'] or v12 != v13['required_v12_spec'] or v13 != spec['required_v13_spec']:
        raise ValueError('The required baseline ensemble specifications differ.')
    rank_factor = v12_extension / 'models' / v12['rank_factor_model']
    verify_hash(rank_factor, v12['rank_factor_sha256'])
    for filename, digest in spec['factor_sha256'].items():
        verify_hash(root / 'models' / filename, digest)
    old_heads = [torch.load(root / 'models' / name, map_location='cpu', weights_only=False)
                 for name in v11['calibration_models']]
    sequence_heads = []
    for filename in v13['sequence_models']:
        path = v13_extension / 'models' / filename
        verify_hash(path, v13['sequence_model_sha256'][filename])
        sequence_heads.append(torch.load(path, map_location='cpu', weights_only=False))
    rate_heads = []
    for filename in spec['rate_models']:
        path = extension / 'models' / filename
        verify_hash(path, spec['rate_model_sha256'][filename])
        rate_heads.append(torch.load(path, map_location='cpu', weights_only=False))
    ids, legacy = predict_v8(root, threads)
    gc.collect()
    train = dict(np.load(root / 'data/train.npz'))
    events = dict(np.load(root / 'data/evaluation.npz'))
    audit = json.loads((root / 'data/audit.json').read_text())
    factor_paths = [root / 'models' / name for name in v11['factor_models']]
    raw, names = build(train, events, spec['cutoff'], audit['n_items'], audit['n_users'], factor_paths[0],
                       feature_state=root / 'models/feature_state.npz',
                       exposure_basis=root / 'data/exposure_basis.npz')
    if any(head['names'] != names for head in old_heads):
        raise ValueError('Calibration feature schema mismatch.')
    if any(head['base_spec']['names'] != names for head in sequence_heads + rate_heads):
        raise ValueError('History/rate feature schema mismatch.')
    old = v11['factor_weights'][0] * np.mean([predict_head(h, raw) for h in old_heads], axis=0)
    update_features(raw, events, factor_paths[1])
    old += v11['factor_weights'][1] * np.mean([predict_head(h, raw) for h in old_heads], axis=0)
    update_features(raw, events, factor_paths[2])
    state = torch.load(factor_paths[2], map_location='cpu', weights_only=False)['state']
    pairs, mask = history_pairs(state, events)
    rates = rate_features(train, events, spec['cutoff'], audit['n_users'])
    del train
    gc.collect()
    rate = np.mean([predict_rate(h, raw, pairs, mask, rates) for h in rate_heads], axis=0)
    raw[:, 225:245] = posterior_features(state, events, samples=spec['posterior_samples'])[0]
    old += v11['factor_weights'][2] * np.mean([predict_head(h, raw) for h in old_heads], axis=0)
    sequence = np.mean([predict_sequence(h, raw, pairs, mask) for h in sequence_heads], axis=0)
    del pairs, mask
    update_features(raw, events, rank_factor)
    state = torch.load(rank_factor, map_location='cpu', weights_only=False)['state']
    raw[:, 225:245] = posterior_features(state, events, samples=spec['posterior_samples'])[0]
    rank = np.mean([predict_head(h, raw) for h in old_heads], axis=0)
    precise = .375*legacy + .15625*old + .15625*rank + .3125*sequence
    alternate = .375*legacy + .625*rate
    weight = float(spec['rate_candidate_weight'])
    if not 0 <= weight <= 1:
        raise ValueError('Rate candidate weight must be between zero and one.')
    probability = (1-weight)*precise + weight*alternate
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert probability.reshape(-1, 10).sum(axis=1).max() <= 1 + 1e-12
    return ids, probability


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='Extracted V11 runtime.')
    parser.add_argument('--v12-extension', required=True)
    parser.add_argument('--v13-extension', required=True)
    parser.add_argument('--extension', required=True, help='Extracted V14 model extension.')
    parser.add_argument('--out', default='P3_Labs_submission_v14.csv')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    ids, probability = predict(args.root, args.v12_extension, args.v13_extension, args.extension, args.threads)
    pd.DataFrame({'rowid': ids, 'predicted_probability': probability}).to_csv(args.out, index=False, float_format='%.10g')
    report = validate(args.out, ids)
    assert report['reference_order_matches']
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
