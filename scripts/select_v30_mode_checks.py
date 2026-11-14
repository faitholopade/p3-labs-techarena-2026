"""Add recipe-specific weekday checks without relaxing the initial V30 gates."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from flexible_tree import predict as legacy_tree_predict
from screen_v30_pooled import datasets,position,atomic_npz
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['weekdays','weekends','train-data','incumbent','screen','v20','old-tree','amendment','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);ds,align=datasets(a,False)
    initial=json.loads((a.screen/'report.json').read_text());old=np.load(a.screen/'controls/v27_predictions.npz');weekend_oldtree=np.load(a.old_tree)
    ev=dict(np.load(a.incumbent));assert np.array_equal(ev['indices'],weekend_oldtree['indices'])
    v20=json.loads((a.v20/'models/v20_spec.json').read_text());tree=next(h for h in v20['heads'] if h['name']=='expanded700')
    treepred={};sources=[]
    for k in range(2):
        al=align[k];pos=position(al);pred=np.full(10*len(pos),np.nan)
        for fold in range(3):
            source=tree['validation_fold_models'][fold];modelpath=a.v20/source['file'];assert sha256(modelpath)==source['sha256'];model=lgb.Booster(model_file=str(modelpath))
            at=np.flatnonzero(al['fold'][pos]==fold)
            for start in range(0,len(at),2000):
                ix=at[start:start+2000];rows=(pos[ix,None]*10+np.arange(10)).ravel();dest=(ix[:,None]*10+np.arange(10)).ravel()
                pred[dest]=legacy_tree_predict(model,ds[k]['x'][rows],tree['training_spec'])
            sources.append({'cohort':ds[k]['name'],'fold':fold,'model_sha256':source['sha256']})
        assert np.isfinite(pred).all();treepred[ds[k]['name']]=pred
    atomic_npz(a.out/'legacy_tree_weekday_predictions.npz',**treepred)
    predictions={n:np.load(a.screen/n/'predictions.npz') for n in initial['plan']['variants']};rows=[];saved={}
    for row in initial['results']:
        r=dict(row);checks={};new=predictions[r['variant']];w=r['weight'];mode=r['mode']
        for k in range(2):
            name=ds[k]['name'];al=align[k];pos=position(al);y=al['labels'][(pos[:,None]*10+np.arange(10)).ravel()]
            baseline=treepred[name] if mode=='replace_old_tree' else old[name]
            candidate=(1-w)*baseline+w*new[name] if mode=='blend' else new[name]
            checks[name]={'baseline_auc':float(roc_auc_score(y,baseline)),'candidate_auc':float(roc_auc_score(y,candidate)),
                          'gain':float(roc_auc_score(y,candidate)-roc_auc_score(y,baseline))}
        r['mode_weekday_checks']=checks;r['mode_weekday_checks_pass']=all(c['gain']>0 for c in checks.values());rows.append(r)
        if mode=='replace_old_tree':candidate=ev['candidate']+w*(new['weekend']-weekend_oldtree['prediction'])
        elif mode=='replace_pooled_head':candidate=ev['candidate']+w*(new['weekend']-old['weekend'])
        else:candidate=(1-w)*ev['candidate']+w*new['weekend']
        actual=roc_auc_score(ev['labels'],candidate)-roc_auc_score(ev['labels'],ev['candidate']);assert abs(actual-r['gains']['combined'])<1e-12
        saved[r['name']]=candidate
    plan=initial['plan'];eligible=[r for r in rows if r['both_weekend_dates_improve'] and r['both_weekday_head_checks_improve'] and r['mode_weekday_checks_pass'] and r['logloss_change']<=0 and r['gains']['combined']>=plan['minimum_exploratory_gain']]
    selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible);near=[r for r in eligible if r['gains']['combined']>=best-plan['tie_tolerance']]
        selected=min(near,key=lambda r:(r['mode']=='blend',r['weight'],-r['gains']['combined']))
        atomic_npz(a.out/'evidence.npz',indices=ev['indices'],labels=ev['labels'],baseline=ev['candidate'],candidate=saved[selected['name']],period=ev['period'],users=ev['users'])
    report={'initial_report_sha256':sha256(a.screen/'report.json'),'amendment':json.loads(a.amendment.read_text()),'source_sha256':sha256(Path(__file__)),
            'legacy_tree_sources':sources,'results':rows,'selected':selected,'initial_selection':initial['selected'],'portal_uploads':0,
            'limits':'Recipe-specific weekday component diagnostics tighten the initial gate. They do not reconstruct the full incumbent on weekdays; all diagnostic cohorts were previously used in research.'}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
