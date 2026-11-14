"""Run the frozen query-slate prior experiment; does not use portal attempts."""
import argparse,hashlib,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from common import load_data,subset,binary_labels
from query_slate_prior import fit_prior,correction,adjust
from rebuild_benchmark import runtime_path,restore,sha,write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','benchmark','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--plan',type=Path,default=REPO/'results/v41_experiment_plan.json');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());inputs,meta=restore(a.benchmark);d,_,_=load_data(runtime_path(a.runtime,11)/'data')
    frozen={'plan_sha256':sha(a.plan),'data_sha256':sha(runtime_path(a.runtime,11)/'data/train.npz'),'source_sha256':{str(x.relative_to(REPO)):sha(x) for x in [Path(__file__),REPO/'src/query_slate_prior.py']}}
    if (a.out/'inputs.json').exists():raise ValueError('Use a fresh experiment directory')
    write(a.out/'inputs.json',frozen);controls={};deltas={};checks=[]
    matched=dict(np.load(inputs/'evidence/matched_v38.npz'))
    cohorts=[('thursday',1737590400),('friday',1737676800),('earlier',1737763200),('later',1737849600)]
    for label,cut in cohorts:
        if label in ['thursday','friday']:
            ev=dict(np.load(inputs/f'evidence/{label}_components.npz'));factor=inputs/f'factors/{label}.pt'
        else:
            per=0 if label=='earlier' else 1;mask=matched['period']==per
            ev={k:v[mask] for k,v in matched.items() if k!='indices'};ev['indices']=matched['indices'][mask.reshape(-1,10)[:,0]]
            factor=runtime_path(a.runtime,17)/f'validation/factors/{label}_factor_users_only.pt'
        state=torch.load(factor,map_location='cpu',weights_only=False);assert state['cut']==cut
        fit=subset(d,np.flatnonzero(d['timestamp']<cut));query=subset(d,ev['indices']);assert np.array_equal(binary_labels(query).ravel(),ev['labels'])
        query['target']=np.full(len(query['user']),-1);assert fit['timestamp'].max()<cut<=query['timestamp'].min()
        started=time.time();prior=fit_prior(state['state'],fit,**{'alpha':plan['ridge']['alpha'],'minimum_events':plan['ridge']['minimum_prior_events']})
        np.savez_compressed(a.out/f'{label}_prior.npz',**prior);controls[label]=ev;deltas[label]={}
        for name in plan['variants']:deltas[label][name]=correction(state['state'],prior,query,shrink=name=='variance_shrunk')
        small=subset(query,np.arange(min(64,len(query['user']))));changed={k:v.copy() for k,v in small.items()};changed['target'][:]=0
        assert np.array_equal(correction(state['state'],prior,small),correction(state['state'],prior,changed))
        changed['items']=changed['items'][:,::-1].copy()
        permutation_error=float(np.max(abs(correction(state['state'],prior,small)[:,::-1]-correction(state['state'],prior,changed))))
        assert permutation_error<1e-12
        assert np.array_equal(adjust(ev['candidate'],deltas[label]['variance_shrunk'],0),ev['candidate'])
        checks.append({'cohort':label,'factor_sha256':sha(factor),'dense_users':prior['dense_users'],'seconds':time.time()-started,'query_target_invariant':True,'permutation_error':permutation_error,'zero_strength_exact':True})
        np.savez_compressed(a.out/f'{label}_deltas.npz',**deltas[label]);print(checks[-1],flush=True)
    y=matched['labels'];base=matched['candidate'];masks={'combined':np.ones(len(y),bool),'earlier':matched['period']==0,'later':matched['period']==1}
    base_auc={k:float(roc_auc_score(y[m],base[m])) for k,m in masks.items()};results=[];predictions={}
    for name in plan['variants']:
        delta=np.concatenate([deltas[label][name] for label in ['earlier','later']])
        for strength in plan['strengths']:
            candidate=adjust(base,delta,strength);assert candidate.reshape(-1,10).sum(1).max()<=1+1e-12
            gains={k:float(roc_auc_score(y[m],candidate[m])-base_auc[k]) for k,m in masks.items()}
            loss=float(log_loss(y,candidate)-log_loss(y,base));weekday={}
            for label in ['thursday','friday']:
                ev=controls[label];cp=adjust(ev['candidate'],deltas[label][name],strength)
                weekday[label]=float(roc_auc_score(ev['labels'],cp)-roc_auc_score(ev['labels'],ev['candidate']))
            eligible=gains['combined']>=1e-5 and gains['earlier']>0 and gains['later']>0 and loss<=0 and all(v>0 for v in weekday.values())
            key=f'{name}_{strength}';predictions[key]=candidate
            results.append({'name':key,'variant':name,'strength':strength,'gains':gains,'logloss_change':loss,'weekday_component_gains':weekday,'eligible':eligible})
    eligible=[r for r in results if r['eligible']];selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible)
        selected=min([r for r in eligible if r['gains']['combined']>=best-2e-6],key=lambda r:(r['variant']!='variance_shrunk',r['strength'],-r['gains']['combined']))
        np.savez_compressed(a.out/'selected_evidence.npz',indices=matched['indices'],labels=y,users=matched['users'],period=matched['period'],baseline=base,candidate=predictions[selected['name']])
    report={'plan':plan,'inputs':frozen,'checks':checks,'baseline_auc':base_auc,'results':results,'selected':selected,'official_auc':None,'portal_uploads':0}
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
