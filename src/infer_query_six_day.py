"""Reproduce scored V51 and apply a fixed six-day correction recipe."""
import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from common import load_data
from infer_v20 import predict as parent_predict
from infer_v38 import serialize, convex_rebuild, sha, canonical_sha
from calibration_model import predict as head_predict
from v25_exposure_memory import features as memory_features
from v30_tree import predict as tree_predict
from validate_submission import validate
from query_context_features import ContextView, NAMES as CONTEXT_NAMES
from linear_calibration import predict as linear_predict
from bounded_history import predict as history_predict
from sequence_calibration import history_pairs
from user_response import summarize as response_summaries, predict as response_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ['runtime', 'spec', 'out']:
        parser.add_argument('--' + key, type=Path, required=True)
    parser.add_argument('--repo', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--cache', type=Path, help='Optional new verified V38 cache; the damaged V21 cache is incompatible.')
    parser.add_argument('--threads', type=int, default=3)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    assert spec['version'] in ['v54', 'v55', 'v57', 'v58', 'v59', 'v61', 'v63']
    v51_path = args.repo / 'artifacts/v51/spec.json'
    assert sha(v51_path) == spec['required_v51_spec_sha256']
    v51 = json.loads(v51_path.read_text())
    v45_path = args.repo / 'artifacts/v45/spec.json'
    assert sha(v45_path) == spec['required_v45_spec_sha256']
    v45 = json.loads(v45_path.read_text())
    assert sha(Path(__file__)) == spec['source_sha256']['src/infer_query_six_day.py']
    for name, expected in spec['source_sha256'].items():
        if sha(args.repo / name) != expected:
            raise ValueError('Source changed: ' + name)
    for name, expected in spec['dependency_canonical_sha256'].items():
        if canonical_sha(args.repo / name) != expected:
            raise ValueError('Dependency changed: ' + name)
    v38_path = args.repo / 'artifacts/v38/spec.json'
    assert sha(v38_path) == spec['required_v38_spec_sha256']
    v38 = json.loads(v38_path.read_text())
    v33_path = args.repo / 'artifacts/v33/spec.json'
    assert sha(v33_path) == v38['required_v33_spec_sha256']
    v33 = json.loads(v33_path.read_text())

    def runtime(version):
        original = args.runtime / ('P3_Labs_v11_model_and_code' if version == 11 else f'P3_Labs_v{version}_models')
        compact = args.runtime / f'v{version}'
        if original.is_dir():
            return original
        if compact.is_dir():
            return compact
        raise FileNotFoundError(f'Missing runtime v{version}')

    runtimes = {version: runtime(version) for version in [11, 12, 13, 15, 16, 17, 18, 19, 20]}
    root = runtimes[11]
    for name, expected in v33['data_sha256'].items():
        if sha(root / 'data' / name) != expected:
            raise ValueError('Dataset changed: ' + name)
    if args.cache:
        for name, expected in spec['cache_sha256'].items():
            if sha(args.cache / name) != expected:
                raise ValueError('Narrow-head cache mismatch: ' + name)
        with zipfile.ZipFile(args.cache / 'components.npz') as archive:
            assert archive.testzip() is None
        cached = dict(np.load(args.cache / 'components.npz'))
        ids, base = cached['rowid'], cached['v20']
        raw = np.load(args.cache / 'raw.npy', mmap_mode='r')
        names = json.loads((args.cache / 'names.json').read_text())
    else:
        ids, base, raw, names, cached = parent_predict(
            *[runtimes[v] for v in [11, 12, 13, 15, 16, 17, 18, 19, 20]],
            threads=args.threads, return_components=True)
    reference = np.load(root / 'data/evaluation_rowids.npy')
    assert len(reference) == 529390 and np.array_equal(ids, reference)
    components = {'rowid': ids, 'v20': base, 'v17': cached['v17'],
                  'expanded700': cached['expanded700'], 'neural32': cached['neural32']}
    parent_hashes = {}

    def verify_parent(name, prediction):
        actual = hashlib.sha256(serialize(ids, prediction).encode()).hexdigest()
        expected = spec['required_v38_csv_sha256'] if name == 'v38' else v38['required_' + name + '_csv_sha256']
        if actual != expected:
            raise ValueError('Scored parent failed to reproduce: ' + name)
        parent_hashes[name] = actual

    verify_parent('v20', base)
    assert np.max(abs(base - (.765625 * components['v17'] + .109375 * components['expanded700'] + .125 * components['neural32']))) < 5e-16
    fit, query, _ = load_data(root / 'data')
    query['target'] = np.full(len(query['user']), -1)
    factor = runtimes[17] / 'models/factor_users_only.pt'
    assert sha(factor) == v33['memory_factor_sha256']
    state = torch.load(factor, map_location='cpu', weights_only=False)
    assert state['cut'] == 1737936000
    extra, extra_names = memory_features(fit, query, state['state'], 1737936000)
    memory = np.c_[raw, extra]
    memory_names = names + extra_names

    def checked_path(entry):
        path = (args.repo / entry['path']).resolve()
        if not path.is_relative_to(args.repo.resolve()) or sha(path) != entry['sha256']:
            raise ValueError('Invalid model artifact: ' + entry['path'])
        return path

    for entry in v33['heads']:
        model = torch.load(checked_path(entry), map_location='cpu', weights_only=False)
        data, schema = (raw, names) if entry['feature_set'] == 'base' else (memory, memory_names)
        assert model['names'] == schema
        components[entry['name']] = head_predict(model, data)
    verify_parent('v27', .75 * base + .25 * components['v27_head'])
    verify_parent('v29', .75 * base + .25 * components['v29_head'])
    verify_parent('v31', .75 * base + .125 * components['v27_head'] + .125 * components['v29_head'])
    parent = .65625 * base + .109375 * components['v27_head'] + .109375 * components['v29_head'] + .125 * components['v33_head']
    verify_parent('v33', parent)
    tree = lgb.Booster(model_file=str(checked_path(v38['tree'])))
    tree_spec = json.loads(checked_path(v38['tree_training_spec']).read_text())
    assert tree_spec['names'] == names
    components['v38_tree'] = np.concatenate([tree_predict(tree, raw[start:start + 20000], tree_spec)
                                             for start in range(0, len(raw), 20000)])
    slate = torch.load(checked_path(v38['slate_head']), map_location='cpu', weights_only=False)
    assert slate['names'] == memory_names and slate['architecture'] == 'set'
    components['v38_slate'] = head_predict(slate, memory)
    baseline = .875 * (parent + .07177734375 * (components['v38_tree'] - components['expanded700'])) + .125 * components['v38_slate']
    verify_parent('v38', baseline)
    components['v38_baseline'] = baseline
    assert np.max(abs(baseline - convex_rebuild(components))) < 5e-16
    for entry in v45['new_heads']:
        model = torch.load(checked_path(entry), map_location='cpu', weights_only=False)
        assert model['names'] == memory_names and model['train_events'] == 588310
        assert model['seed'] == entry['seed'] and model['architecture'] == entry['architecture']
        assert model['width'] == entry['width'] and model['memory_drop_probability'] == 0
        components[entry['name']] = head_predict(model, memory)
    mode = v45['recipe']['mode']
    small = components.get('plain_20261029', components['v33_head'])
    slate = components.get('slate_20261029', components['v38_slate'])
    changed = dict(components)
    candidate = baseline.copy()
    if mode in ['replace_small', 'replace_both']:
        candidate += .109375 * (small - components['v33_head'])
        changed['v33_head'] = small
    if mode in ['replace_slate', 'replace_both']:
        candidate += .125 * (slate - components['v38_slate'])
        changed['v38_slate'] = slate
    if mode == 'blend':
        weight = v45['recipe']['weight']
        candidate = (1 - weight) * baseline + weight * (small + slate) / 2
        rebuilt = (1 - weight) * convex_rebuild(components) + weight * (small + slate) / 2
    else:
        rebuilt = convex_rebuild(changed)
    v45_sha = hashlib.sha256(serialize(ids, candidate).encode()).hexdigest()
    assert v45_sha == spec['required_v45_csv_sha256']
    parent_hashes['v45'] = v45_sha
    components['v45_baseline'] = candidate.copy()
    narrow_model = torch.load(checked_path(v51['new_head']), map_location='cpu', weights_only=False)
    assert narrow_model['names'] == memory_names and narrow_model['train_events'] == 588310
    assert narrow_model['seed'] == 20261029 and narrow_model['architecture'] == 'plain'
    assert narrow_model['width'] == 16 and narrow_model['memory_drop_probability'] == 0
    narrow = head_predict(narrow_model, memory)
    components['narrow_20261029'] = narrow
    for key, branch in [('base', 'v27_head'), ('memory', 'v29_head')]:
        fraction = v51['recipe'][key]; assert fraction == .5
        candidate += fraction * .095703125 * (narrow - components[branch])
        changed[branch] = (1 - fraction) * components[branch] + fraction * narrow
    v51_sha = hashlib.sha256(serialize(ids, candidate).encode()).hexdigest()
    assert v51_sha == spec['required_v51_csv_sha256']
    parent_hashes['v51'] = v51_sha
    components['v51_baseline'] = candidate.copy()
    new_model = torch.load(checked_path(spec['new_head']), map_location='cpu', weights_only=False)
    new_names = memory_names + CONTEXT_NAMES if spec['version'] == 'v54' else memory_names
    assert new_model['names'] == new_names and new_model['train_events'] == spec['training_events']
    assert new_model['seed'] == 20261029 and new_model['architecture'] == spec['architecture']
    assert new_model['width'] == spec['width'] and new_model['memory_drop_probability'] == 0
    new_features = ContextView(memory, query['context']) if spec['version'] == 'v54' else memory
    if spec['architecture'] in ['bounded_history', 'bounded_user_response']:
        assert new_model['base_frozen']
        frozen = new_model['base_spec']
        for key in ['selected', 'names', 'width', 'architecture', 'offset_column']:
            assert frozen[key] == narrow_model[key]
        for key in ['median', 'mean', 'std']:
            assert np.array_equal(frozen[key], narrow_model[key])
        assert all(torch.equal(value, narrow_model['state'][key]) for key, value in frozen['state'].items())
        if spec['architecture'] == 'bounded_history':
            pair, history_mask = history_pairs(state['state'], query)
            context_head = history_predict(new_model, new_features, pair, history_mask)
        else:
            summaries = response_summaries(state['state'], fit, 1737936000)
            context_head = response_predict(new_model, new_features, summaries[query['user']])
    else:
        context_head = (linear_predict if spec['architecture'] == 'linear' else head_predict)(new_model, new_features)
    components['query_head_20261029'] = context_head
    recipe = spec['recipe']
    for key in ['narrow', 'plain', 'base', 'memory']:
        assert 0 <= recipe[key] <= 1
    candidate += .095703125 * recipe['narrow'] * (context_head - narrow)
    candidate += .109375 * recipe['plain'] * (context_head - small)
    for key, branch in [('base', 'v27_head'), ('memory', 'v29_head')]:
        candidate += .0478515625 * recipe[key] * (context_head - components[branch])
        changed[branch] = (.5 * ((1 - recipe['narrow']) * narrow + recipe['narrow'] * context_head)
                           + .5 * ((1 - recipe[key]) * components[branch] + recipe[key] * context_head))
    changed['v33_head'] = (1 - recipe['plain']) * small + recipe['plain'] * context_head
    rebuilt = convex_rebuild(changed)
    error = float(np.max(abs(candidate - rebuilt)))
    assert error < 5e-16
    assert np.isfinite(candidate).all() and candidate.min() >= 0 and candidate.max() <= 1
    assert candidate.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
    components.update(candidate=candidate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p3_context_', dir=args.out.parent) as temporary:
        staged = Path(temporary) / args.out.name
        staged.write_text(serialize(ids, candidate))
        component_path = staged.with_suffix('.components.npz')
        np.savez_compressed(component_path, **components)
        with zipfile.ZipFile(component_path) as archive:
            assert archive.testzip() is None
        report = validate(staged, reference)
        assert report['reference_order_matches']
        mass = float(pd.read_csv(staged).predicted_probability.to_numpy().reshape(-1, 10).sum(1).max())
        assert mass <= 1 + 1e-8
        report.update(version=spec['version'], spec_sha256=sha(args.spec), source_sha256=sha(Path(__file__)),
                      csv_sha256=sha(staged), scored_parent_hashes=parent_hashes, convex_max_abs_error=error,
                      max_event_probability=mass, components_crc_valid=True, atomic_staged_inference=True,
                      release_mode=spec['release_mode'], normal_quality_gate=spec['normal_quality_gate'], official_score=None,
                      portal_uploads=0, inference_mode='Verified rebuilt V38 cache' if args.cache else 'Fresh parent inference')
        validation_path = staged.with_suffix('.validation.json')
        validation_path.write_text(json.dumps(report, indent=2) + '\n')
        for source, target in [(staged, args.out), (component_path, args.out.with_suffix('.components.npz')),
                               (validation_path, args.out.with_suffix('.validation.json'))]:
            expected = sha(source)
            if target.exists() and sha(target) != expected:
                raise ValueError('Refusing to overwrite different output: ' + target.name)
            os.replace(source, target)
            assert sha(target) == expected
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
