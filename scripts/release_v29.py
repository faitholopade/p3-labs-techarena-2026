"""Freeze the selected, fully trained V29 head and its convex inference recipe."""
import argparse,json,shutil,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--research',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--destination',type=Path,default=REPO/'artifacts/v29');a=p.parse_args()
    report=json.loads((a.research/'screen/report.json').read_text());selected=report['selected']
    if not selected:raise ValueError('No selected candidate.')
    assessment=json.loads((a.research/'assessment/decision.json').read_text())
    if assessment['incumbent']!='v27' or not assessment['both_dates_improve']:raise ValueError('Wrong or regressing baseline.')
    if not assessment['positive_bootstrap_lower_bound']:raise ValueError('Uncertainty gate failed; do not release automatically.')
    name=selected['variant'];config=report['plan']['variants'][name]
    full=a.research/'screen'/name/'full';meta=json.loads((full/'report.json').read_text());model=full/'model.pt'
    if meta['model_sha256']!=sha256(model) or meta['training_events']!=391710:raise ValueError('Full checkpoint failed validation.')
    if meta['plan']['config']!=config:raise ValueError('Full-fit configuration differs from selection.')
    a.destination.mkdir(parents=True,exist_ok=True);dest=a.destination/'head.pt'
    if dest.exists() and sha256(dest)!=sha256(model):raise ValueError('Refusing to replace a different release checkpoint.')
    shutil.copyfile(model,dest)
    old=json.loads((REPO/'artifacts/v27/spec.json').read_text());w=float(selected['weight'])
    if selected['mode']=='replace_pooled_head':
        if w!=.25:raise ValueError('Unexpected replacement weight.')
        base_weight=.75;old_weight=.25-w;new_weight=w
    elif selected['mode']=='blend':base_weight=.75*(1-w);old_weight=.25*(1-w);new_weight=w
    else:raise ValueError('Unknown mode.')
    if config['head_average_with_v27']:old_weight+=new_weight/2;new_weight/=2
    heads=[]
    if old_weight:
        heads.append({'name':'v27_head','path':'artifacts/v27/pooled_head.pt','sha256':old['heads'][0]['sha256'],
                      'weight':old_weight,'feature_set':'base'})
    heads.append({'name':'v29_head','path':str(dest.relative_to(REPO)),'sha256':sha256(dest),
                  'weight':new_weight,'feature_set':'memory' if config['columns']==388 else 'base'})
    spec={'version':'v29','variant':name,'official_auc':None,'selected':selected,'v20_weight':base_weight,'heads':heads,
          'required_v20_csv_sha256':old['required_v20_csv_sha256'],'data_sha256':old['data_sha256'],
          'cache_sha256':old['cache_sha256'],'training_events':391710,'training_config':config,
          'evidence_sha256':sha256(a.research/'screen/selected_evidence.npz'),
          'experiment_plan_sha256':sha256(a.research/'plan.json'),
          'source_sha256':{'inference':sha256(REPO/'src/infer_v29.py'),'training':sha256(REPO/'src/daily_training.py')},
          'decision':'Normal quality gate passed' if assessment['normal_quality_gate'] else 'Standing exploratory authorization; normal gate not met',
          'portal_uploads':0}
    if config['columns']==388:
        paths=[a.runtime/'P3_Labs_v17_models/models/factor_users_only.pt',a.runtime/'v17/models/factor_users_only.pt']
        factor=next((p for p in paths if p.exists()),None)
        if factor is None:raise FileNotFoundError('V17 memory factor missing.')
        spec['memory_factor_sha256']=sha256(factor)
    destspec=a.destination/'spec.json'
    if destspec.exists() and json.loads(destspec.read_text())!=spec:raise ValueError('Release specification is frozen.')
    write_json(destspec,spec);print(json.dumps(spec),flush=True)

if __name__=='__main__':main()
