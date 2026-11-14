"""Freeze a selected V35 architecture after all precommitted checks pass."""
import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from backtest_v24_factor import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research', type=Path, required=True)
    a = p.parse_args()
    selected = json.loads((a.research / 'screen/report.json').read_text())['selected']
    if not selected or not selected['eligible_before_confirmation']:
        raise ValueError('No eligible screened recipe.')
    quality = json.loads((a.research / 'assessment/decision.json').read_text())
    if quality['incumbent'] != 'v33' or not quality['both_dates_improve'] or not quality['positive_bootstrap_lower_bound']:
        raise ValueError('Paired uncertainty check failed.')
    variant = selected['variant']
    forward = json.loads((a.research / f'screen/{variant}/forward/report.json').read_text())
    recipe = {'mode': selected['mode'], 'weight': selected['weight']}
    if forward['diagnostic']['gain'] < 0 or forward['diagnostic']['recipe'] != recipe:
        raise ValueError('Forward diagnostic failed or used a different recipe.')
    weights = {'replace_small_head': [.65625, .109375, .109375, 0., .125],
               'replace_pooled_heads': [.65625, 0., 0., 0., .34375],
               'blend': [.57421875, .095703125, .095703125, .109375, .125]}[selected['mode']]
    if selected['weight'] != weights[-1]:
        raise ValueError('Unexpected mixture weight.')
    full = a.research / f'screen/{variant}/full'
    report = json.loads((full / 'report.json').read_text())
    if report['training_events'] != 391710 or report['model_sha256'] != sha256(full / 'model.pt'):
        raise ValueError('Full model differs from its completed report.')
    plan = json.loads((REPO / 'results/v35_experiment_plan.json').read_text())
    if report['plan']['config'] != plan['variants'][variant]:
        raise ValueError('Full model configuration differs from the frozen plan.')
    old = json.loads((REPO / 'artifacts/v33/spec.json').read_text())
    out = REPO / 'artifacts/v35'; out.mkdir(parents=True, exist_ok=True)
    head = out / 'head.pt'
    if head.exists() and sha256(head) != report['model_sha256']:
        raise ValueError('Refusing to replace a different full model.')
    shutil.copyfile(full / 'model.pt', head)
    heads = []
    for entry, weight in zip(old['heads'], weights[1:4]):
        if sha256(REPO / entry['path']) != entry['sha256']:
            raise ValueError('An existing released head changed.')
        heads.append(dict(entry, weight=weight))
    heads.append({'name': 'v35_head', 'path': 'artifacts/v35/head.pt',
                  'sha256': sha256(head), 'weight': weights[-1], 'feature_set': 'memory'})
    spec = {'version': 'v35', 'created_on': '2026-10-29', 'official_auc': None,
            'incumbent': 'v33', 'official_incumbent_auc': .832616,
            'variant': variant, 'recipe': recipe, 'v20_weight': weights[0], 'heads': heads,
            'required_v33_csv_sha256': '29843af0d37c56ed095306002e01d44abcda9b52bc78ae7d94a2708e17180507',
            'training_events': report['training_events'],
            'evidence_sha256': sha256(a.research / 'screen/selected_evidence.npz'),
            'plan_sha256': sha256(REPO / 'results/v35_experiment_plan.json'),
            'selection_sha256': sha256(a.research / 'screen/report.json'),
            'forward_sha256': sha256(full.parent / 'forward/report.json'),
            'source_sha256': {name: sha256(REPO / path) for name, path in
                              [('wrapper', 'src/infer_v35.py'), ('base_inference', 'src/infer_v29.py'),
                               ('trainer', 'src/daily_training.py'), ('model', 'src/calibration_model.py'),
                               ('memory_features', 'src/v25_exposure_memory.py')]},
            'release_mode': 'normal_quality_gate' if quality['normal_quality_gate'] else 'user_authorized_exploratory',
            'portal_uploads': 0}
    for key in ['required_v20_csv_sha256', 'required_v27_csv_sha256', 'required_v29_csv_sha256',
                'required_v31_csv_sha256', 'data_sha256', 'cache_sha256', 'memory_factor_sha256']:
        spec[key] = old[key]
    path = out / 'spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Release recipe is frozen.')
    write_json(path, spec)
    print(json.dumps(spec), flush=True)


if __name__ == '__main__':
    main()
