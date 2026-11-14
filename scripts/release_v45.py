"""Freeze the explicitly exploratory V45 release after completed assessment and fits."""
import argparse
import json
import shutil
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import torch
from rebuild_benchmark import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    args = parser.parse_args()
    release = json.loads((args.work / 'release_decision.json').read_text())
    selected = release['selected']; assert selected['recipe']['name'] == 'replace_both'
    original = json.loads((args.work / 'report.json').read_text()); assert original['selected'] is None
    assessment = json.loads((args.work / 'assessment/paired_bootstrap.json').read_text())
    gates = json.loads((args.work / 'assessment/decision.json').read_text())
    forward = json.loads((args.work / 'forward_assessment/report.json').read_text())
    v38 = json.loads((REPO / 'artifacts/v38/spec.json').read_text())
    models = []
    for family, architecture in [('plain', 'plain'), ('slate', 'set')]:
        source = args.work / family / 'full/model.pt'
        completed = json.loads(source.with_name('model_completed.json').read_text())
        assert sha(source) == completed['sha256'] and completed['training_events'] == 588310
        model = torch.load(source, map_location='cpu', weights_only=False)
        assert model['width'] == 32 and model['architecture'] == architecture and model['seed'] == 20261029
        path = f'artifacts/v45/{family}_head.pt'; (REPO / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, REPO / path)
        models.append({'name': family + '_20261029', 'path': path, 'sha256': sha(source),
                       'seed': 20261029, 'width': 32, 'architecture': architecture, 'training_events': 588310})
        for stage in ['folds', 'forward', 'full']:
            shutil.copyfile(args.work / family / stage / 'report.json', REPO / f'results/v45_{family}_{stage}.json')
    for source, destination in [('report.json', 'v45_report.json'), ('release_decision.json', 'v45_release_decision.json'),
            ('assessment/paired_bootstrap.json', 'v45_paired_bootstrap.json'), ('assessment/decision.json', 'v45_assessment_decision.json'),
            ('forward_assessment/report.json', 'v45_forward_diagnostic.json')]:
        shutil.copyfile(args.work / source, REPO / 'results' / destination)
    spec = {'version': 'v45', 'created_on': '2026-10-30', 'recipe': selected['recipe'],
            'incumbent': 'v38', 'official_incumbent_auc': .832634, 'official_score': None, 'portal_uploads': 0,
            'release_mode': release['release_mode'], 'authorization': release['authorization'],
            'formula': 'V38 + 0.109375*(six_day_plain-old_small) + 0.125*(six_day_slate-old_slate)',
            'new_heads': models, 'training_events': 588310,
            'required_v38_spec_sha256': sha(REPO / 'artifacts/v38/spec.json'),
            'required_v38_csv_sha256': '95ef7daab9ec97ff6260d1d195bd28889e63f6c3b20b1cbbfafe7a02a56b38b3',
            'source_sha256': {p: sha(REPO / p) for p in ['src/infer_v45.py', 'src/infer_v38.py']},
            'dependency_canonical_sha256': v38['dependency_canonical_sha256'],
            'training_source_sha256': {p: sha(REPO / p) for p in ['scripts/prepare_v45_calibration.py',
                'scripts/research_v45.py', 'scripts/run_v45_compact_folds.py', 'scripts/run_v45_compact_forward.py',
                'scripts/fit_v45_full.py', 'src/v33_training.py', 'src/v45_training.py', 'src/v45_compact_training.py']},
            'plan_sha256': sha(REPO / 'results/v45_experiment_plan.json'),
            'control_clarification_sha256': sha(REPO / 'results/v45_control_clarification.json'),
            'selection_sha256': sha(args.work / 'report.json'), 'release_decision_sha256': sha(args.work / 'release_decision.json'),
            'evidence_sha256': sha(args.work / 'selected_evidence.npz'),
            'local_gain_vs_v38': selected['gains']['combined'], 'local_gains': selected['gains'],
            'weekday_component_gains': selected['weekday_component_gains'],
            'conditional_interval_95_percent': assessment['interval_95_percent'],
            'forward_component_gain': forward['gain'], 'forward_sha256': sha(args.work / 'forward_assessment/report.json'),
            'frozen_screen_passed': False, 'normal_quality_gate': gates['normal_quality_gate'],
            'scarce_slot_quality_gate': gates['last_two_quality_gate'],
            'positive_bootstrap_lower_bound': gates['positive_bootstrap_lower_bound'],
            'cache_sha256': {name: sha(args.cache / name) for name in ['raw.npy', 'names.json', 'components.npz']},
            'cache_provenance': 'Verified rebuilt V38 cache; fresh parent inference reproduced the scored V38 CSV byte for byte.',
            'limits': 'Below frozen +0.000050 screen and normal/scarce release gates. Conditional interval crosses zero and excludes selection and distribution-shift uncertainty. Weekday/forward diagnostics omit unchanged historical components. Replacement of the small branch also changes width from 16 to 32. No public gain or rank guarantee.'}
    path = REPO / 'artifacts/v45/spec.json'
    if path.exists():
        assert json.loads(path.read_text()) == spec
    write(path, spec)
    print(json.dumps({'release_frozen': True, 'spec_sha256': sha(path), 'models': len(models),
                      'local_gain': spec['local_gain_vs_v38'], 'normal_quality_gate': spec['normal_quality_gate']}), flush=True)


if __name__ == '__main__':
    main()
