"""Freeze a selected six-day recipe after paired and chronological confirmation."""
import argparse
import json
import shutil
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from rebuild_benchmark import sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='Selected candidate evidence directory.')
    parser.add_argument('--models-work', type=Path, required=True)
    parser.add_argument('--version', choices=['v54', 'v55', 'v57', 'v58', 'v59', 'v61', 'v63'], required=True)
    parser.add_argument('--exploratory', action='store_true', help='Record the standing user-authorized exception if standard quality gates fail.')
    args = parser.parse_args()
    report = json.loads((args.work / 'report.json').read_text())
    selected = report['selected']
    assert selected is not None and selected['eligible'], 'No eligible frozen recipe.'
    forward = json.loads((args.work / 'forward_report.json').read_text())
    decision = json.loads((args.work / 'assessment/decision.json').read_text())
    uncertainty = json.loads((args.work / 'assessment/paired_bootstrap.json').read_text())
    assert forward['recipe'] == selected['recipe'] and forward['gain'] > 0
    assert forward['logloss_change'] <= 0 and decision['both_dates_improve']
    assert args.exploratory or decision['normal_quality_gate'], 'Explicit exploratory switch required.'
    assert abs(report['baseline_auc'] - .8378178361843427) < 1e-12
    completed = json.loads((args.models_work / 'full/model_completed.json').read_text())
    assert sha(args.models_work / 'full/model.pt') == completed['sha256']
    inputs = json.loads((args.models_work / 'full/model_inputs.json').read_text())
    assert inputs['plan_sha256'] == report['plan_sha256'] == sha(REPO / f'results/{args.version}_experiment_plan.json')
    assert inputs['sources'][f'scripts/research_{args.version}.py'] == report['source_sha256']
    expected_events = sum(inputs['features']['first_query_training_events']) if args.version in ['v55', 'v63'] else 588310
    assert completed['training_events'] == expected_events
    old_path = REPO / 'artifacts/v51/spec.json'; old = json.loads(old_path.read_text())
    old_verification = json.loads((REPO / 'results/v51_submission_verification.json').read_text())
    assert sha(old_path) == old_verification['spec_sha256']
    folder = REPO / 'artifacts' / args.version; folder.mkdir(parents=True, exist_ok=True)
    source, target = args.models_work / 'full/model.pt', folder / 'query_head.pt'
    if target.exists(): assert sha(target) == sha(source)
    else: shutil.copy2(source, target)
    entries = {'new_head': {'path': str(target.relative_to(REPO)), 'sha256': sha(target)}}
    sources = ['src/infer_query_six_day.py', 'src/user_response.py', 'src/first_query_adaptation.py', 'src/linear_calibration.py', 'src/bounded_history.py', 'src/sequence_calibration.py', 'src/query_context_features.py', 'src/infer_narrow_six_day.py', 'src/infer_v45.py', 'src/infer_v38.py']
    spec = dict(version=args.version, created_on='2026-10-30', incumbent='v51', official_incumbent_auc=.832677,
                official_score=None, portal_uploads=0, recipe=selected['recipe'],
                formula='V51 plus fixed transfers from the selected narrow/plain/remaining-old branches into the selected query head; weights 0.095703125, 0.109375, 0.0478515625, 0.0478515625.',
                release_mode='explicit_user_requested_exploratory_attempt' if not decision['normal_quality_gate'] else 'standard_quality_gate',
                authorization='Standing request to prepare the strongest supported next exploratory submission, renewed after official V51. Manual portal upload remains with Faith.',
                normal_quality_gate=decision['normal_quality_gate'], last_two_quality_gate=decision['last_two_quality_gate'],
                positive_bootstrap_lower_bound=decision['positive_bootstrap_lower_bound'],
                required_v51_spec_sha256=sha(old_path), required_v51_csv_sha256=old_verification['csv_sha256'],
                required_v45_spec_sha256=old['required_v45_spec_sha256'], required_v45_csv_sha256=old['required_v45_csv_sha256'],
                required_v38_spec_sha256=old['required_v38_spec_sha256'], required_v38_csv_sha256=old['required_v38_csv_sha256'],
                source_sha256={name: sha(REPO / name) for name in sources},
                dependency_canonical_sha256=old['dependency_canonical_sha256'], cache_sha256=old['cache_sha256'],
                training_events=expected_events, feature_mode={'v54': 'context', 'v55': 'first_query', 'v57': 'linear', 'v58': 'plain8', 'v59': 'bounded_history', 'v61': 'bounded_user_response', 'v63': 'first_query_adaptation'}[args.version],
                architecture=report['plan'].get('architecture', 'plain'), width=report['plan']['training']['width'], local_gains=selected['gains'], logloss_change=selected['logloss_change'],
                weekday_component_gains=selected['weekday_component_gains'], forward_component_gain=forward['gain'],
                conditional_interval_95_percent=uncertainty['interval_95_percent'],
                plan_sha256=report['plan_sha256'], research_source_sha256=report['source_sha256'],
                limits='Repeated development use. Other-user calibration dates are pooled. Weekday and forward checks omit unchanged V17/V20 components. Bootstrap is conditional on the selected recipe and excludes selection and public-distribution uncertainty. No promised public gain or rank.', **entries)
    path = folder / 'spec.json'
    if path.exists(): assert json.loads(path.read_text()) == spec, 'Released specifications are immutable.'
    else: write(path, spec)
    for source, suffix in [('report.json', 'screen_report.json'), ('forward_report.json', 'forward_diagnostic.json'),
                           ('assessment/decision.json', 'assessment_decision.json'), ('assessment/paired_bootstrap.json', 'paired_bootstrap.json'),
                           ('assessment/report.json', 'assessment_report.json')]:
        shutil.copy2(args.work / source, REPO / 'results' / (args.version + '_' + suffix))
    shutil.copy2(args.models_work / 'full/model_report.json', REPO / 'results' / (args.version + '_full_fit.json'))
    release = dict(incumbent='v51', selected=selected, confirmation=decision, forward_component_gain=forward['gain'],
                   release_mode=spec['release_mode'], spec_sha256=sha(path), official_score=None, portal_uploads=0,
                   authorization=spec['authorization'], failed_standard_gates=[k for k in ['normal_quality_gate', 'last_two_quality_gate', 'positive_bootstrap_lower_bound'] if not decision[k]])
    write(REPO / 'results' / (args.version + '_release_decision.json'), release)
    print(json.dumps(release), flush=True)


if __name__ == '__main__': main()
