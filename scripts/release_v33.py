"""Freeze a V33 recipe only after the precommitted checks pass."""
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
    screen = json.loads((a.research / 'screen/report.json').read_text())
    selected = screen['selected']
    if not selected or not selected['eligible_before_confirmation']:
        raise ValueError('No eligible screened recipe.')
    quality = json.loads((a.research / 'assessment/decision.json').read_text())
    if quality['incumbent'] != 'v31' or not quality['both_dates_improve'] or not quality['positive_bootstrap_lower_bound']:
        raise ValueError('Paired uncertainty check failed.')
    variant = selected['variant']
    forward = json.loads((a.research / f'screen/{variant}/forward/report.json').read_text())
    if forward['diagnostic']['gain'] < 0:
        raise ValueError('Forward diagnostic failed.')
    recipe = {'mode': selected['mode'], 'weight': selected['weight']}
    if forward['diagnostic']['recipe'] != recipe:
        raise ValueError('Forward diagnostic evaluated a different recipe.')
    weights = {'replace_memory': [.75, .125, 0., .125],
               'replace_pooled_heads': [.75, 0., 0., .25],
               'blend': [.65625, .109375, .109375, .125]}[selected['mode']]
    if selected['weight'] != weights[-1]:
        raise ValueError('Unexpected mixture weight.')
    full = a.research / f'screen/{variant}/full'
    report = json.loads((full / 'report.json').read_text())
    if report['training_events'] != 391710 or report['model_sha256'] != sha256(full / 'model.pt'):
        raise ValueError('Full model differs from its completed report.')
    old = json.loads((REPO / 'artifacts/v31/spec.json').read_text())
    out = REPO / 'artifacts/v33'; out.mkdir(parents=True, exist_ok=True)
    head = out / 'head.pt'
    if head.exists() and sha256(head) != report['model_sha256']:
        raise ValueError('Refusing to replace a different full model.')
    shutil.copyfile(full / 'model.pt', head)
    heads = []
    for entry, weight in zip(old['heads'], weights[1:3]):
        if sha256(REPO / entry['path']) != entry['sha256']:
            raise ValueError('Existing released head changed.')
        heads.append(dict(entry, weight=weight))
    heads.append({'name': 'v33_head', 'path': 'artifacts/v33/head.pt',
                  'sha256': sha256(head), 'weight': weights[-1], 'feature_set': 'memory'})
    spec = {'version': 'v33', 'created_on': '2026-10-29', 'official_auc': None,
            'incumbent': 'v31', 'official_incumbent_auc': .832599,
            'variant': variant, 'recipe': recipe, 'v20_weight': weights[0], 'heads': heads,
            'required_v31_csv_sha256': 'e27d314e3da4d5f895d404a2e1450893e7b68116a5464066d72cdc5b426dd5e8',
            'training_events': report['training_events'],
            'evidence_sha256': sha256(a.research / 'screen/selected_evidence.npz'),
            'plan_sha256': sha256(REPO / 'results/v33_experiment_plan.json'),
            'selection_sha256': sha256(a.research / 'screen/report.json'),
            'forward_sha256': sha256(full.parent / 'forward/report.json'),
            'source_sha256': {name: sha256(REPO / path) for name, path in
                              [('wrapper', 'src/infer_v33.py'), ('base_inference', 'src/infer_v29.py'),
                               ('trainer', 'src/v33_training.py')]},
            'release_mode': 'user_authorized_exploratory', 'portal_uploads': 0}
    for key in ['required_v20_csv_sha256', 'required_v27_csv_sha256', 'required_v29_csv_sha256',
                'data_sha256', 'cache_sha256', 'memory_factor_sha256']:
        spec[key] = old[key]
    path = out / 'spec.json'
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError('Release recipe is frozen.')
    write_json(path, spec)
    print(json.dumps(spec), flush=True)


if __name__ == '__main__':
    main()
