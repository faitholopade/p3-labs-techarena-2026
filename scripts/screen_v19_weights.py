"""Disclosed small weight-transfer grid using existing matched components."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();r=a.root
    ev=dict(np.load(r/'runtime/P3_Labs_v18_models/validation/evidence.npz'));v16=np.load(r/'runtime/P3_Labs_v16_models/validation/evidence.npz');v17=np.load(r/'runtime/P3_Labs_v17_models/validation/evidence.npz');e12=np.load(r/'runtime/P3_Labs_v12_models/validation/v12/evidence.npz');legacy=np.load(r/'experiments/v18/cache/legacy.npy')
    assert np.array_equal(ev['labels'],e12['labels']) and np.array_equal(ev['indices'],e12['indices'])
    members=[]
    for member,screen in [('cold_kl1','screen'),('users_only','screen256')]:
        values=np.empty(len(ev['labels']));seen=np.zeros(len(values),bool)
        for period in ['earlier','later']:
            d=np.load(r/f'experiments/v17/{period}_{member}/{screen}/predictions.npz');rows=(d['selection'][:,None]*10+np.arange(10)).ravel();assert np.array_equal(d['labels'],ev['labels'][rows]) and not seen[rows].any();values[rows]=d['sequence'];seen[rows]=True
        assert seen.all();members.append(values)
    sequence=np.mean(members,axis=0);parent=.75*v16['candidate']+.09375*legacy+.15625*sequence
    error=float(np.max(np.abs(parent-v17['candidate'])));assert error<1e-12
    oldtree=np.load(r/'experiments/v18/tree500/predictions.npz')['prediction'];assert np.max(np.abs(.875*parent+.125*oldtree-ev['candidate']))<1e-12
    base=ev['candidate'];masks={'combined':np.ones(len(base),bool),'earlier':ev['period']==0,'later':ev['period']==1};base_auc={k:float(roc_auc_score(ev['labels'][m],base[m])) for k,m in masks.items()};rows=[]
    for donor,values in [('legacy',legacy),('v12',e12['v12'])]:
        for delta in [-.05,-.025,.025,.05,.1]:
            prediction=base+delta*(sequence-values)
            assert prediction.min()>=0 and prediction.max()<=1 and prediction.reshape(-1,10).sum(1).max()<=1+1e-12
            auc={k:float(roc_auc_score(ev['labels'][m],prediction[m])) for k,m in masks.items()};gain={k:auc[k]-base_auc[k] for k in auc}
            rows.append({'donor':donor,'delta':delta,'auc':auc,'gain':gain,'both_dates_improve':gain['earlier']>0 and gain['later']>0})
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'components.npz',legacy=legacy,v12=e12['v12'],sequence=sequence,baseline=base,labels=ev['labels'],indices=ev['indices'])
    report={'baseline':'v18','reconstructed_v17_max_error':error,'candidates':rows,'limits':'Ten additional comparisons on reused development cohorts; these are explicit weight transfers, not new factor fits.'};(a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(rows,indent=2),flush=True)


if __name__=='__main__':main()
