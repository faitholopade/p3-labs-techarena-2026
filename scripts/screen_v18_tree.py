"""Fit an event-choice tree head with the same user-excluded validation folds."""
import argparse,json,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.metrics import roc_auc_score
from event_tree import fit,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=500);p.add_argument('--leaves',type=int,default=15);p.add_argument('--seed',type=int,default=20261102);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--full',action='store_true')
    p.add_argument('--neural-offsets',type=Path,help='Offsets from frozen heads excluding each outer validation fold.')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);x=np.load(a.features/'raw.npy',mmap_mode='r');fold=np.load(a.features/'fold.npy')
    fs=json.loads((a.features/'spec.json').read_text());ev=np.load(a.incumbent);y=ev['labels'];base=ev['candidate'];users=ev['users'].reshape(-1,10)[:,0]
    offsets=np.load(a.neural_offsets) if a.neural_offsets else None
    if a.full:
        model,spec=fit(x,y,fs['offset_column'],a.rounds,a.leaves,a.seed,a.threads,external_offset=None if offsets is None else offsets['full']);model.save_model(str(a.out/'tree.txt'))
        spec['names']=json.loads((a.features/'names.json').read_text());spec['feature_spec']=fs
        (a.out/'spec.json').write_text(json.dumps(spec,indent=2));print({'full_rows':len(y)},flush=True);return
    pred=np.empty(len(y));records=[]
    for f in range(3):
        train_events=np.flatnonzero(fold!=f);test_events=np.flatnonzero(fold==f)
        assert not np.intersect1d(users[train_events],users[test_events]).size
        tr=(train_events[:,None]*10+np.arange(10)).ravel();te=(test_events[:,None]*10+np.arange(10)).ravel()
        offset=None if offsets is None else offsets[f'fold{f}']
        model,spec=fit(x[tr],y[tr],fs['offset_column'],a.rounds,a.leaves,a.seed,a.threads,external_offset=None if offset is None else offset[tr]);pred[te]=predict(model,x[te],spec,external_offset=None if offset is None else offset[te])
        model.save_model(str(a.out/f'fold_{f}.txt'));records.append({'fold':f,'train_events':len(train_events),'test_events':len(test_events),'users_disjoint':True});print(records[-1],flush=True)
        del model;gc.collect()
    results=[]
    for w in [.125,.25,.5,1.]:
        blended=(1-w)*base+w*pred;result={'weight':w,'gains':{},'auc':{}}
        for name,mask in [('combined',np.ones(len(y),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc=float(roc_auc_score(y[mask],blended[mask]));result['auc'][name]=auc;result['gains'][name]=auc-float(roc_auc_score(y[mask],base[mask]))
        results.append(result)
    np.savez_compressed(a.out/'predictions.npz',prediction=pred,baseline=base,labels=y,indices=ev['indices'])
    report={'training_spec':spec,'folds':records,'results':results,'base_version':'v17','limitation':'Reused development cohorts; no held-out user labels in that fold head.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
