"""Record the explicit V38 exploration exception and lock the two full models."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
from backtest_v24_factor import sha256,write_json
from infer_v38 import canonical_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research',type=Path,required=True)
    p.add_argument('--v36',type=Path,required=True);a=p.parse_args()
    plan=json.loads((REPO/'results/v38_release_plan.json').read_text())
    screen=json.loads((a.v36/'screen/report.json').read_text());quality=json.loads((a.v36/'assessment/decision.json').read_text())
    interval=json.loads((a.v36/'assessment/paired_bootstrap.json').read_text())
    assert screen['selected']['recipe']=='tree_plus_slate' and quality['incumbent']=='v33'
    assert quality['both_dates_improve'] and not quality['positive_bootstrap_lower_bound']
    assert plan['release_mode']=='explicit_user_requested_exploratory_attempt'
    assert interval['auc_improvement']==plan['known_local_gain']
    assert interval['interval_95_percent']==plan['known_conditional_interval_95_percent']
    forward=json.loads((a.research/'fits/forward_report.json').read_text())
    assert forward['recipe']=='tree_plus_slate'
    out=REPO/'artifacts/v38';out.mkdir(parents=True,exist_ok=True);entries={};training={}
    for kind,destination in [('tree','tree.txt'),('slate','slate_head.pt')]:
        folder=a.research/'fits'/(kind+'_full');report=json.loads((folder/'report.json').read_text())
        source=folder/report['model_file'];assert sha256(source)==report['model_sha256'] and report['training_events']==391710
        assert report['inputs']['plan_sha256']==sha256(REPO/'results/v38_release_plan.json')
        assert report['inputs']['config']==plan[kind+'_config']
        target=out/destination
        if target.exists() and sha256(target)!=sha256(source):raise ValueError('Full release model changed.')
        shutil.copyfile(source,target);entries[kind]={'path':str(target.relative_to(REPO)),'sha256':sha256(target)}
        training[kind]=report
    tree_spec=out/'tree_training_spec.json';shutil.copyfile(a.research/'fits/tree_full/training_spec.json',tree_spec)
    dependencies=['src/v30_tree.py','src/flexible_tree.py','src/calibration_model.py','src/v25_exposure_memory.py','src/validate_submission.py']
    old=json.loads((REPO/'artifacts/v33/spec.json').read_text())
    spec={'version':'v38','created_on':'2026-10-29','incumbent':'v33','official_incumbent_auc':.832616,
          'recipe':'tree_plus_slate','tree_replacement_weight':.07177734375,'slate_weight':.125,
          'formula':plan['formula'],'tree':entries['tree'],'slate_head':entries['slate'],
          'tree_training_spec':{'path':str(tree_spec.relative_to(REPO)),'sha256':sha256(tree_spec)},
          'required_v33_spec_sha256':sha256(REPO/'artifacts/v33/spec.json'),
          'required_v33_csv_sha256':'29843af0d37c56ed095306002e01d44abcda9b52bc78ae7d94a2708e17180507',
          'source_sha256':{'src/infer_v38.py':sha256(REPO/'src/infer_v38.py')},
          'dependency_canonical_sha256':{p:canonical_sha(REPO/p) for p in dependencies},
          'plan_sha256':sha256(REPO/'results/v38_release_plan.json'),
          'evidence_sha256':sha256(a.v36/'screen/selected_evidence.npz'),
          'selection_sha256':sha256(a.v36/'screen/report.json'),
          'forward_sha256':sha256(a.research/'fits/forward_report.json'),
          'training_events':391710,'training_reports_sha256':{k:sha256(a.research/'fits'/(k+'_full')/'report.json') for k in training},
          'release_mode':plan['release_mode'],'authorization':plan['authorization'],
          'local_gain_vs_v33':interval['auc_improvement'],'conditional_interval_95_percent':interval['interval_95_percent'],
          'positive_bootstrap_lower_bound':False,'normal_quality_gate':False,'scarce_slot_quality_gate':False,
          'forward_component_gain':forward['gain'],'official_score':None,'portal_uploads':0,'limits':plan['limits']}
    for key in ['required_v20_csv_sha256','required_v27_csv_sha256','required_v29_csv_sha256','required_v31_csv_sha256']:
        spec[key]=old[key]
    path=out/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Release specification changed.')
    write_json(path,spec);print(json.dumps(spec),flush=True)


if __name__=='__main__':main()
