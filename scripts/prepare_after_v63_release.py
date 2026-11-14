"""Freeze an eligible, confirmed V65/V66/V67 recipe without changing V63."""
import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'scripts'))
from rebuild_benchmark import sha

V63_SPEC_SHA = '791a30996f504740d8a77cfa4c3ff5938f7ad9b42418da2f50b321b7fbaa8530'
V63_CSV_SHA = '22f19ffd8c7cc7cde828c3fdfe2effdacf5232e6738dbde139e1eaeac8eef850'
V63_LOCAL_AUC = .8378351482608294


def read(path):
    return json.loads(path.read_text())


def immutable_copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(target) == sha(source), 'Refusing to change released file: ' + str(target)
    else:
        shutil.copy2(source, target)


def immutable_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert read(path) == value, 'Released records are immutable: ' + str(path)
    else:
        path.write_text(json.dumps(value, indent=2) + '\n')


def matching_gains(left, right):
    assert set(left) == set(right) == {'combined', 'earlier', 'later'}
    assert all(abs(left[k] - right[k]) < 1e-12 for k in left), 'Assessment and selected recipe differ.'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True, help='Selected screen/forward/assessment directory.')
    parser.add_argument('--models-work', type=Path, required=True,
                        help='V65 work directory, or V66 work directory for V66/V67 models.')
    parser.add_argument('--version', choices=['v65', 'v66', 'v67'], required=True)
    parser.add_argument('--exploratory', action='store_true',
                        help='Record the standing user-authorized exception when the normal gate fails.')
    args = parser.parse_args()
    version = args.version
    report = read(args.work / 'report.json')
    selected = report['selected']
    assert selected is not None and selected['eligible'], 'No eligible frozen winner.'
    recipe = selected['recipe']
    plan_path = REPO / f'results/{version}_experiment_plan.json'
    plan = read(plan_path)
    runner = f'scripts/research_{version}.py'
    assert report['plan'] == plan and report['plan_sha256'] == sha(plan_path)
    assert report['source_sha256'] == sha(REPO / runner), 'Screen source changed.'
    assert recipe in plan['recipes'], 'Selected recipe was not fixed in the experiment plan.'
    assert selected in report['results']
    eligible = [row for row in report['results'] if row['eligible']]
    best_gain = max(row['gains']['combined'] for row in eligible)
    contenders = [row for row in eligible if row['gains']['combined'] >= best_gain - .000005]
    tie_key = (lambda row: row['recipe']['strength']) if version == 'v65' else (lambda row: row['total_changed_weight'])
    assert selected == min(contenders, key=tie_key), 'Selected recipe violates the fixed selection rule.'
    assert abs(report['baseline_auc'] - V63_LOCAL_AUC) < 1e-12
    assert selected['gains']['combined'] >= .000010
    assert min(selected['gains'].values()) > 0 and selected['logloss_change'] <= 0
    assert set(selected['weekday_component_gains']) == {'thursday', 'friday'}
    assert min(selected['weekday_component_gains'].values()) > 0

    forward = read(args.work / 'forward_report.json')
    decision = read(args.work / 'assessment/decision.json')
    uncertainty = read(args.work / 'assessment/paired_bootstrap.json')
    assessment = read(args.work / 'assessment/report.json')
    assert forward['recipe'] == recipe and forward['gain'] > 0
    assert forward['logloss_change'] <= 0 and forward['events'] == 35040
    assert decision['candidate'] == version and decision['incumbent'] == 'v63'
    matching_gains(selected['gains'], decision['local_gains'])
    assert decision['both_dates_improve']
    assert uncertainty['replicates'] >= 400 and uncertainty['seed'] == 20261030
    assert abs(uncertainty['auc_improvement'] - selected['gains']['combined']) < 1e-12
    assert decision['positive_bootstrap_lower_bound'] == (uncertainty['interval_95_percent'][0] > 0)
    for key in ['combined', 'earlier', 'later']:
        gains = assessment[key][version]['auc'] - assessment[key]['v63']['auc']
        assert abs(gains - selected['gains'][key]) < 1e-12
    assert abs(assessment['combined']['v63']['auc'] - V63_LOCAL_AUC) < 1e-12
    policy = read(REPO / 'config/submission_policy.json')
    for key, minimum in [('normal_quality_gate', 'normal_minimum_local_gain'),
                         ('last_two_quality_gate', 'scarce_slots_minimum_local_gain')]:
        expected = selected['gains']['combined'] >= policy[minimum] and decision['positive_bootstrap_lower_bound']
        assert decision[key] == expected
    assert args.exploratory or decision['normal_quality_gate'], 'Explicit --exploratory required.'

    parent_path = REPO / 'artifacts/v63/spec.json'
    verification = read(REPO / 'results/v63_submission_verification.json')
    assert sha(parent_path) == verification['spec_sha256'] == V63_SPEC_SHA
    assert verification['csv_sha256'] == V63_CSV_SHA
    parent = read(parent_path)
    v51 = read(REPO / 'artifacts/v51/spec.json')
    v45 = read(REPO / 'artifacts/v45/spec.json')
    assert sha(REPO / 'artifacts/v51/spec.json') == parent['required_v51_spec_sha256']
    assert sha(REPO / 'artifacts/v45/spec.json') == parent['required_v45_spec_sha256']
    for name, expected in parent['source_sha256'].items():
        assert sha(REPO / name) == expected, 'Pinned V63 source changed: ' + name

    fit_version = 'v65' if version == 'v65' else 'v66'
    fit_plan_path = REPO / f'results/{fit_version}_experiment_plan.json'
    fit_plan = read(fit_plan_path)
    if version == 'v67':
        assert report['model_plan_sha256'] == sha(fit_plan_path), 'V67 must reference the frozen V66 fit plan.'
    families = ['event'] if version == 'v65' else [k for k in ['plain', 'slate'] if recipe[k] > 0]
    assert families, 'Release must include at least one new model.'
    sources = dict(parent['source_sha256'])
    for name in ['src/infer_after_v63.py', 'scripts/prepare_after_v63_release.py', runner]:
        sources[name] = sha(REPO / name)
    sources['src/event_mass_correction.py' if version == 'v65' else 'src/first_query_adaptation.py'] = sha(
        REPO / ('src/event_mass_correction.py' if version == 'v65' else 'src/first_query_adaptation.py'))
    models = []
    provenance = {}
    model_files = []
    for family in families:
        folder = args.models_work / ('full' if family == 'event' else family + '/full')
        path = folder / 'model.pt'
        completed = read(folder / 'model_completed.json')
        inputs = read(folder / 'model_inputs.json')
        fit_report = read(folder / 'model_report.json')
        assert sha(path) == completed['sha256']
        assert inputs['plan_sha256'] == sha(fit_plan_path)
        assert inputs['stage'] == 'full' and inputs['fold'] is None
        assert fit_report['inputs'] == inputs and fit_report['fit'] == completed
        assert f'scripts/research_{fit_version}.py' in inputs['sources']
        for name, expected in inputs['sources'].items():
            assert sha(REPO / name) == expected, 'Full-fit source changed: ' + name
            assert name not in sources or sources[name] == expected
            sources[name] = expected
        if family == 'event':
            expected_events, width, architecture = 588310, 16, 'event_mass'
            assert inputs['base_model_sha256'] == v51['new_head']['sha256']
        else:
            assert inputs['family'] == family and completed['family'] == family
            expected_events = sum(inputs['features']['first_query_training_events'])
            assert expected_events == 210588
            width, architecture = 32, 'plain' if family == 'plain' else 'set'
            prior = next(e for e in v45['new_heads'] if e['name'] == family + '_20261029')
            assert inputs['base_model_sha256'] == prior['sha256']
        assert completed['training_events'] == expected_events
        model = torch.load(path, map_location='cpu', weights_only=False)
        assert model['train_events'] == expected_events and model['width'] == width
        assert model['architecture'] == architecture and model['seed'] == fit_plan['seed']
        if family == 'event':
            assert model['base_frozen']
        else:
            assert model['adapted_first_queries'] and model['initial_normalization_preserved']
            assert model['initial_training_events'] == 588310
        del model
        target = REPO / f'artifacts/{version}/{family}_head.pt'
        entry = dict(path=str(target.relative_to(REPO)), sha256=sha(path), family=family,
                     architecture=architecture, width=width, seed=fit_plan['seed'], training_events=expected_events)
        models.append(entry)
        provenance[family] = dict(fit_version=fit_version, plan_path=str(fit_plan_path.relative_to(REPO)),
                                 plan_sha256=sha(fit_plan_path), model_inputs_sha256=sha(folder / 'model_inputs.json'),
                                 model_completed_sha256=sha(folder / 'model_completed.json'),
                                 model_report_sha256=sha(folder / 'model_report.json'), inputs=inputs, completed=completed)
        model_files.append((path, target))

    evidence_files = [('report.json', 'screen_report.json'), ('forward_report.json', 'forward_diagnostic.json'),
                      ('assessment/decision.json', 'assessment_decision.json'),
                      ('assessment/paired_bootstrap.json', 'paired_bootstrap.json'),
                      ('assessment/report.json', 'assessment_report.json')]
    evidence_hashes = {f'results/{version}_{suffix}': sha(args.work / source)
                       for source, suffix in evidence_files}
    formula = ('V63 conditional item probabilities scaled by the event mass sigmoid(logit(V63 mass) + strength * bounded delta).'
               if version == 'v65' else 'V63 + 0.109375 * plain_fraction * (adapted_plain32 - V45_plain32) + 0.125 * slate_fraction * (adapted_slate32 - V45_slate32) + 0.095703125 * narrow_fraction * (V63_adapted_plain16 - V50_plain16); narrow_fraction is zero for V66.')
    spec = dict(version=version, created_on='2026-10-30', incumbent='v63', official_incumbent_auc=.832698,
                official_score=None, portal_uploads=0, recipe=recipe, formula=formula, new_models=models,
                release_mode='standard_quality_gate' if decision['normal_quality_gate'] else 'explicit_user_requested_exploratory_attempt',
                authorization='Standing user request to prepare the strongest supported next exploratory submission, renewed after official V63. Manual portal upload remains with Faith.',
                normal_quality_gate=decision['normal_quality_gate'], last_two_quality_gate=decision['last_two_quality_gate'],
                positive_bootstrap_lower_bound=decision['positive_bootstrap_lower_bound'],
                required_v63_spec_sha256=V63_SPEC_SHA, required_v63_csv_sha256=V63_CSV_SHA,
                source_sha256=sources, dependency_canonical_sha256=parent['dependency_canonical_sha256'],
                cache_sha256=parent['cache_sha256'], plan_sha256=sha(plan_path), research_source_sha256=report['source_sha256'],
                fit_provenance=provenance, evidence_sha256=evidence_hashes,
                selected_evidence_sha256=sha(args.work / 'selected_evidence.npz'),
                forward_evidence_sha256=sha(args.work / 'forward_evidence.npz'),
                local_gains=selected['gains'], logloss_change=selected['logloss_change'],
                weekday_component_gains=selected['weekday_component_gains'], forward_component_gain=forward['gain'],
                forward_logloss_change=forward['logloss_change'], conditional_interval_95_percent=uncertainty['interval_95_percent'],
                limits='Reused development cohorts and selected-recipe conditional uncertainty. Other-user calibration dates are pooled. Weekday and chronological diagnostics omit unchanged V17/V20 components. No promised public gain or rank.')
    spec_path = REPO / f'artifacts/{version}/spec.json'
    if spec_path.exists():
        assert read(spec_path) == spec, 'Released specifications are immutable.'
    for source, target in model_files:
        immutable_copy(source, target)
    for source, suffix in evidence_files:
        immutable_copy(args.work / source, REPO / 'results' / f'{version}_{suffix}')
    for family in families:
        folder = args.models_work / ('full' if family == 'event' else family + '/full')
        immutable_copy(folder / 'model_report.json', REPO / 'results' / f'{version}_{family}_full_fit.json')
    immutable_json(spec_path, spec)
    release = dict(incumbent='v63', selected=selected, confirmation=decision,
                   forward_component_gain=forward['gain'], forward_logloss_change=forward['logloss_change'],
                   release_mode=spec['release_mode'], spec_sha256=sha(spec_path), official_score=None, portal_uploads=0,
                   authorization=spec['authorization'], failed_standard_gates=[key for key in
                   ['normal_quality_gate', 'last_two_quality_gate', 'positive_bootstrap_lower_bound'] if not decision[key]])
    immutable_json(REPO / 'results' / f'{version}_release_decision.json', release)
    print(json.dumps(release), flush=True)


if __name__ == '__main__':
    main()
