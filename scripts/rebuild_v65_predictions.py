"""Rebuild the scored V65 development baseline from its shared saved models."""
import argparse
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from bounded_history import base_logits
from event_mass_correction import features, delta, transfer
from rebuild_benchmark import sha, write
from research_v43 import positions

V65_SPEC_SHA = '8c5999e12869378eb803e9199999c87b9765ede2b55c9c2aa3214b2ece8c8691'
V65_MATCHED_AUC = .8378525735530878
DAYS = ['thursday', 'friday', 'weekend']


def preserve_or_write(path, arrays):
    """Check existing evidence without rewriting its original ZIP bytes."""
    if path.exists():
        with zipfile.ZipFile(path) as archive:
            assert archive.testzip() is None, str(path)
        with np.load(path) as existing:
            assert set(existing.files) == set(arrays), str(path)
            for key, value in arrays.items():
                assert np.array_equal(existing[key], value), f'{path}: {key}'
        return dict(existing=True, arrays_exact=True, sha256=sha(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as target:
        temporary = Path(target.name)
    try:
        np.savez_compressed(temporary, **arrays)
        with zipfile.ZipFile(temporary) as archive:
            assert archive.testzip() is None
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(existing=False, arrays_exact=True, sha256=sha(path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['benchmark', 'v63', 'out']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    spec_path = REPO / 'artifacts/v65/spec.json'
    assert sha(spec_path) == V65_SPEC_SHA
    spec = json.loads(spec_path.read_text())
    assert spec['recipe'] == dict(name='event_strength_1.0', strength=1.0)
    assert sha(REPO / 'artifacts/v63/spec.json') == spec['required_v63_spec_sha256']
    for name, expected in spec['source_sha256'].items():
        assert sha(REPO / name) == expected, name
    plan_path = REPO / 'results/v65_experiment_plan.json'
    assert sha(plan_path) == spec['plan_sha256']
    model_index = json.loads((REPO / 'artifacts/v65/research/index.json').read_text())
    expected = {row['path']: row['sha256'] for row in model_index['models']}
    narrow_index = json.loads((REPO / 'artifacts/v50/research/index.json').read_text())
    narrow_expected = {row['restore_to']: row['sha256'] for row in narrow_index['models']}
    frozen = json.loads((REPO / 'artifacts/v65/research/folds/fold_0_inputs.json').read_text())
    raw, alignment, names, hashes = {}, {}, {}, {}
    for day in DAYS:
        folder = args.benchmark / 'features' / day
        hashes[day] = {name: sha(folder / name) for name in ['raw.npy', 'alignment.npz', 'names.json']}
        assert hashes[day] == frozen['features'][day], day
        raw[day] = np.load(folder / 'raw.npy', mmap_mode='r')
        alignment[day] = dict(np.load(folder / 'alignment.npz'))
        names[day] = json.loads((folder / 'names.json').read_text())
    model_hashes = {}

    def load_model(stage, fold=None):
        stem = f'fold_{fold}' if fold is not None else 'model'
        relative = f'artifacts/v65/research/{stage}/{stem}.pt'
        path = REPO / relative
        model_hashes[relative] = sha(path)
        assert model_hashes[relative] == expected[relative]
        inputs = json.loads(path.with_name(stem + '_inputs.json').read_text())
        completed = json.loads(path.with_name(stem + '_completed.json').read_text())
        assert completed['sha256'] == model_hashes[relative]
        assert inputs['stage'] == stage and inputs['fold'] == fold
        assert inputs['plan_sha256'] == spec['plan_sha256']
        assert inputs['base_model_sha256'] == narrow_expected[f'{stage}/{stem}.pt']
        for day in DAYS:
            assert inputs['features'][day] == hashes[day]
        for name, expected_source in inputs['sources'].items():
            assert sha(REPO / name) == expected_source, name
        model = torch.load(path, map_location='cpu', weights_only=False)
        assert model['architecture'] == 'event_mass' and model['base_frozen']
        assert model['train_events'] == completed['training_events']
        return model

    corrections = {day: np.full(len(positions(al)), np.nan, np.float32)
                   for day, al in alignment.items()}
    for fold in range(3):
        model = load_model('folds', fold)
        for day in DAYS:
            assert names[day] == model['names']
            utility = base_logits(model['base_spec'], raw[day])
            event_features, event_names = features(raw[day], names[day], utility)
            assert event_names == model['event_names']
            pos = positions(alignment[day])
            test = np.flatnonzero(alignment[day]['fold'][pos] == fold)
            assert np.isnan(corrections[day][test]).all()
            corrections[day][test] = delta(model, event_features[pos[test]])
            del utility, event_features
        print(json.dumps(dict(completed_fold=fold)), flush=True)
    assert all(np.isfinite(value).all() for value in corrections.values())
    previous = dict(np.load(args.v63 / 'current_predictions.npz'))
    evidence = dict(np.load(args.v63 / 'selected_evidence.npz'))
    assert np.array_equal(previous['weekend'], evidence['candidate'])
    assert abs(roc_auc_score(evidence['labels'], evidence['candidate']) - .8378351482608294) < 1e-12
    current = {day: transfer(previous[day], corrections[day], 1.0) for day in DAYS}
    auc = float(roc_auc_score(evidence['labels'], current['weekend']))
    assert abs(auc - V65_MATCHED_AUC) < 1e-12
    records = {}
    records['selected_evidence.npz'] = preserve_or_write(
        args.out / 'selected_evidence.npz',
        dict(indices=evidence['indices'], labels=evidence['labels'], users=evidence['users'],
             period=evidence['period'], baseline=evidence['candidate'], candidate=current['weekend']))
    records['current_predictions.npz'] = preserve_or_write(args.out / 'current_predictions.npz', current)
    records['current_event_delta.npz'] = preserve_or_write(args.out / 'current_event_delta.npz', corrections)

    forward_path = args.v63 / 'forward_evidence.npz'
    forward = dict(available=False, reason='V63 forward evidence not supplied; matched baseline is complete.')
    if forward_path.exists():
        model = load_model('forward')
        assert model['names'] == names['friday']
        utility = base_logits(model['base_spec'], raw['friday'])
        event_features, event_names = features(raw['friday'], names['friday'], utility)
        assert event_names == model['event_names']
        pos = positions(alignment['friday'])
        correction = delta(model, event_features[pos])
        source = dict(np.load(forward_path))
        assert np.array_equal(source['indices'], alignment['friday']['indices'][pos])
        candidate = transfer(source['candidate'], correction, 1.0)
        records['forward_evidence.npz'] = preserve_or_write(
            args.out / 'forward_evidence.npz',
            dict(indices=source['indices'], labels=source['labels'], users=source['users'],
                 baseline=source['candidate'], candidate=candidate))
        records['current_forward_delta.npz'] = preserve_or_write(
            args.out / 'current_forward_delta.npz', dict(indices=source['indices'], delta=correction))
        forward = dict(available=True, events=len(pos), auc=float(roc_auc_score(source['labels'], candidate)),
                       scope='First-Friday normalized changed-component diagnostic; historical V17/V20 omitted.')
    report = dict(incumbent='v65', official_auc=.832700, matched_auc=auc, spec_sha256=V65_SPEC_SHA,
                  model_sha256=model_hashes, feature_sha256=hashes, source_sha256=sha(Path(__file__)),
                  outputs=records, forward=forward,
                  scope='Weekend is the complete scored V65 recipe. Thursday/Friday are normalized changed-component diagnostics. Saved models only; no fitting. Existing evidence arrays are checked exactly and their archive bytes preserved.')
    write(args.out / 'baseline_reproduction.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
