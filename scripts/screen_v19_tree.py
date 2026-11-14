"""Compare a fixed tree configuration against the scored V18 incumbent."""
import argparse,json,sys,gc,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.metrics import roc_auc_score
from flexible_tree import fit,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['event','binary','rank_binary'],default='event');p.add_argument('--rounds',type=int,default=500);p.add_argument('--leaves',type=int,default=15)
    p.add_argument('--seed',type=int,default=20261103);p.add_argument('--threads',type=int,default=2);p.add_argument('--full',action='store_true');p.add_argument('--min-leaf',type=int,default=1000)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);raw=np.load(a.features/'raw.npy',mmap_mode='r');fold=np.load(a.features/'fold.npy');names=json.loads((a.features/'names.json').read_text());ev=dict(np.load(a.incumbent));y=ev['labels'];base=ev['candidate'];users=ev['users'].reshape(-1,10)[:,0]
    config={'mode':a.mode,'rounds':a.rounds,'leaves':a.leaves,'seed':a.seed,'threads':a.threads,'min_leaf':a.min_leaf};records=[];start=time.time()
    if a.full:
        model,spec=fit(raw,y,names,config);model.save_model(str(a.out/'tree.txt'));spec['feature_spec']=json.loads((a.features/'spec.json').read_text());spec['training_rows']=len(y)
        (a.out/'spec.json').write_text(json.dumps(spec,indent=2));print({'full_rows':len(y),'seconds':time.time()-start},flush=True);return
    pred=np.empty(len(y))
    for f in range(3):
        train=np.flatnonzero(fold!=f);test=np.flatnonzero(fold==f);assert not np.intersect1d(users[train],users[test]).size
        tr=(train[:,None]*10+np.arange(10)).ravel();te=(test[:,None]*10+np.arange(10)).ravel()
        model,spec=fit(raw[tr],y[tr],names,config);pred[te]=predict(model,raw[te],spec);model.save_model(str(a.out/f'fold_{f}.txt'))
        rec={'fold':f,'train_events':len(train),'test_events':len(test),'users_disjoint':True,'elapsed_seconds':time.time()-start};records.append(rec);print(rec,flush=True);del model;gc.collect()
    np.savez_compressed(a.out/'predictions.npz',prediction=pred,baseline=base,labels=y,indices=ev['indices'])
    results=[]
    for weight in [.0625,.125,.25,.5,1.]:
        blended=(1-weight)*base+weight*pred;gain={};auc={}
        for key,mask in [('combined',np.ones(len(y),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[key]=float(roc_auc_score(y[mask],blended[mask]));gain[key]=auc[key]-float(roc_auc_score(y[mask],base[mask]))
        results.append({'weight':weight,'auc':auc,'gain':gain})
    report={'config':config,'training_spec':spec,'folds':records,'results':results,'base_version':'v18','limitations':'Reused development cohorts; outer calibration folds exclude users across both dates.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
