"""Conditional fixed Thu+Fri calibration transfer to exact V20 weekend rows."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from common import load_data, subset, binary_labels
from v24_causal_propensity import snapshot
from v24_weekday_calibration import build, train, predict, PARAMS, ROUNDS
from v24_item_time import posterior_probability
from screen_v24_weekday_calibration import checked_cohort, digest, THURSDAY, FRIDAY

CUTS=[1737763200,1737849600]
WEIGHTS=[.125,.25,.5]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','thursday-factor','friday-factor','weekday-experiment','v17-extension','incumbent','out']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    factors={period:a.v17_extension/f'validation/factors/{period}_factor_users_only.pt' for period in ['earlier','later']}
    repo=Path(__file__).resolve().parents[1]
    plan={'training_days':[THURSDAY,FRIDAY],'training_label_stop':CUTS[0],
          'params':PARAMS,'rounds':ROUNDS,'weights':WEIGHTS,
          'gate':{'friday_global_auc_gain_strictly_above':.0001,'friday_global_logloss_change_strictly_below':0.},
          'calibration_labels':'Thursday+Friday first-user/day outcomes only; no Saturday or Sunday labels.',
          'offset':'Frozen V17 users-only factor mass, full-scale pure-user posterior; 256 Sobol draws plus antithetic copies.',
          'prediction':'Replace event mass only, preserving exact V20 within-slate item probabilities.',
          'selection':'Three fixed weights; no additional grid or retraining after outcomes.',
          'factor_sha256':{period:digest(path) for period,path in factors.items()},
          'incumbent_sha256':digest(a.incumbent),'data_sha256':digest(a.root/'data/train.npz'),
          'weekday_plan_sha256':digest(a.weekday_experiment/'plan.json'),
          'source_sha256':{s:digest(repo/s) for s in ['src/v24_causal_propensity.py','src/v24_weekday_calibration.py','src/v24_item_time.py','src/posterior_inference.py','scripts/screen_v24_weekday_calibration.py','scripts/transfer_v24_weekday_calibration.py']},
          'scope':'Confirmation on reused matched development dates; no evaluation inference or submission.'}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen transfer plan changed.')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    if a.plan_only:print(json.dumps(plan,indent=2));return
    previous=json.loads((a.weekday_experiment/'report.json').read_text())
    if not (previous['global_auc_gain']>.0001 and previous['global_logloss_change']<0):
        raise ValueError('The fixed Friday transfer gate failed.')
    if (a.out/'model.txt').exists() or (a.out/'report.json').exists():raise ValueError('Existing fitted output; inspect it instead of silently rerunning.')
    started=time.time();data,_,audit=load_data(a.root/'data');torch.set_num_threads(2)
    xs=[];offsets=[];ys=[];training_indices=[];provenance=[];schema=None
    for folder,cut in [(a.thursday_factor,THURSDAY),(a.friday_factor,FRIDAY)]:
        ix,prob,meta=checked_cohort(folder,data,cut)
        s=snapshot(subset(data,np.flatnonzero(data['timestamp']<cut)),audit['n_items'],audit['n_users'],cut)
        query=subset(data,ix);query['target']=np.full(len(ix),-1)
        x,offset,names=build(query,s,prob)
        if schema is not None and schema!=names:raise ValueError('Training feature schema drift.')
        schema=names;xs.append(x);offsets.append(offset);ys.append(data['target'][ix]<10);training_indices.append(ix);provenance.append(meta)
    indices=np.concatenate(training_indices);assert data['timestamp'][indices].max()<CUTS[0]
    model=train(np.concatenate(xs),np.concatenate(offsets),np.concatenate(ys),schema);model.save_model(str(a.out/'model.txt'))
    np.savez_compressed(a.out/'training_alignment.npz',indices=indices,event_labels=np.concatenate(ys))
    ev=dict(np.load(a.incumbent));evi=ev['indices'];period=ev['period'].reshape(-1,10)[:,0]
    assert set(np.unique(period))=={0,1} and np.array_equal(np.repeat(period,10),ev['period'])
    assert np.array_equal(binary_labels(subset(data,evi)).ravel(),ev['labels'])
    rates=np.full(len(evi),np.nan);raw_rates=np.full(len(evi),np.nan);coverage=np.zeros(len(evi),np.int8);period_reports=[]
    for pe,(name,cut) in enumerate(zip(['earlier','later'],CUTS)):
        sel=np.flatnonzero(period==pe);ix=evi[sel];query=subset(data,ix);query['target']=np.full(len(ix),-1)
        assert np.all(data['timestamp'][ix]>=cut)
        ck=torch.load(factors[name],map_location='cpu',weights_only=False)
        if int(ck['cut'])!=cut:raise ValueError('Frozen V17 factor cutoff differs.')
        raw=posterior_probability(ck['state'],query,samples=256)
        s=snapshot(subset(data,np.flatnonzero(data['timestamp']<cut)),audit['n_items'],audit['n_users'],cut)
        x,offset,names=build(query,s,raw);assert names==schema
        calibrated=predict(model,x,offset,raw);new_rate=calibrated.sum(1)
        rates[sel]=new_rate;raw_rates[sel]=raw.sum(1);coverage[sel]+=1
        np.savez_compressed(a.out/f'{name}_factor_predictions.npz',indices=ix,factor_probability=raw,calibrated_probability=calibrated)
        y=data['target'][ix]<10
        period_reports.append({'period':name,'cut':cut,'events':len(ix),'factor_sha256':digest(factors[name]),
                               'max_snapshot_timestamp':s['max_timestamp'],
                               'factor_event_auc':float(roc_auc_score(y,raw.sum(1))),
                               'calibrated_event_auc':float(roc_auc_score(y,new_rate)),
                               'factor_event_logloss':float(log_loss(y,raw.sum(1))),
                               'calibrated_event_logloss':float(log_loss(y,new_rate))})
    assert np.all(coverage==1) and np.isfinite(rates).all() and np.all((rates>=0)&(rates<=1))
    base=ev['candidate'].reshape(-1,10);assert np.all(base.sum(1)>0)
    replacement=base/base.sum(1,keepdims=True)*rates[:,None]
    assert np.allclose(replacement.sum(1),rates,rtol=1e-12,atol=1e-12)
    rows=[]
    for weight in WEIGHTS:
        candidate=((1-weight)*base+weight*replacement).ravel();gain={};auc={}
        for label,m in [('combined',np.ones(len(candidate),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[label]=float(roc_auc_score(ev['labels'][m],candidate[m]));gain[label]=auc[label]-float(roc_auc_score(ev['labels'][m],ev['candidate'][m]))
        row={'weight':weight,'global_auc':auc,'gain_vs_v20':gain,'global_logloss_change':float(log_loss(ev['labels'],candidate)-log_loss(ev['labels'],ev['candidate']))};rows.append(row)
        tag=str(weight).replace('.','p')
        np.savez_compressed(a.out/f'evidence_{tag}.npz',indices=evi,labels=ev['labels'],baseline=ev['candidate'],candidate=candidate,period=ev['period'],user=np.repeat(data['user'][evi],10))
    np.savez_compressed(a.out/'matched_event_predictions.npz',indices=evi,event_probability=rates,raw_factor_event_probability=raw_rates,period=period)
    report={'plan':plan,'training':provenance,'training_events':len(indices),
            'max_training_label_timestamp':int(data['timestamp'][indices].max()),
            'period_diagnostics':period_reports,'results':rows,'model_sha256':digest(a.out/'model.txt'),
            'seconds':time.time()-started,'status':'review_required','no_submission_generated':True,
            'limits':'The calibrator never trains on weekend labels; comparison uses repeatedly reused V20 weekend cohorts and does not estimate public-score gain.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
