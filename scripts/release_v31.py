"""Freeze the single V31 midpoint only after the V30/V31 release comparison."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);a=p.parse_args()
    selection=json.loads(a.selection.read_text());check=json.loads((a.research/'check/report.json').read_text());quality=json.loads((a.research/'assessment/decision.json').read_text())
    if selection['selected_version']!='v31':raise ValueError('V31 was not selected for release.')
    if not check['passes_before_bootstrap'] or not quality['both_dates_improve'] or not quality['positive_bootstrap_lower_bound']:raise ValueError('Exploratory checks failed.')
    old=json.loads((REPO/'artifacts/v27/spec.json').read_text());new=json.loads((REPO/'artifacts/v29/spec.json').read_text())
    for key in ['required_v20_csv_sha256','data_sha256','cache_sha256']:assert old[key]==new[key]
    oldfile=REPO/'artifacts/v27/pooled_head.pt';newfile=REPO/'artifacts/v29/head.pt'
    assert sha256(oldfile)==old['heads'][0]['sha256'] and sha256(newfile)==new['heads'][0]['sha256']
    spec={'version':'v31','created_on':'2026-10-29','official_auc':None,'formula':'0.5*V27+0.5*V29 = 0.75*V20+0.125*V27_head+0.125*V29_memory_head',
          'v20_weight':.75,'heads':[{'name':'v27_head','path':'artifacts/v27/pooled_head.pt','sha256':sha256(oldfile),'weight':.125,'feature_set':'base'},
                                 {'name':'v29_head','path':'artifacts/v29/head.pt','sha256':sha256(newfile),'weight':.125,'feature_set':'memory'}],
          'required_v20_csv_sha256':old['required_v20_csv_sha256'],'required_v27_csv_sha256':'3cda208bb200f6b250a24a9b328cd42999bf801dd64dfb697c28c3aeae2070f6',
          'required_v29_csv_sha256':'21715eadd8a4be4545012fc7769f48c4d3845b1e6825cfde608af402d4949326',
          'data_sha256':old['data_sha256'],'cache_sha256':old['cache_sha256'],'memory_factor_sha256':new['memory_factor_sha256'],
          'new_training_fits':0,'each_pooled_head_training_events':391710,'evidence_sha256':sha256(a.research/'check/evidence.npz'),
          'plan_sha256':sha256(a.research/'plan.json'),'selection_sha256':sha256(a.selection),
          'source_sha256':{'wrapper':sha256(REPO/'src/infer_v31.py'),'base_inference':sha256(REPO/'src/infer_v29.py')},
          'decision':'Standing exploratory authorization below unchanged normal gain threshold; matched and weekday/forward checks pass.','portal_uploads':0}
    out=REPO/'artifacts/v31';out.mkdir(parents=True,exist_ok=True);path=out/'spec.json'
    if path.exists() and json.loads(path.read_text())!=spec:raise ValueError('Release recipe is frozen.')
    write_json(path,spec);print(json.dumps(spec),flush=True)

if __name__=='__main__':main()
