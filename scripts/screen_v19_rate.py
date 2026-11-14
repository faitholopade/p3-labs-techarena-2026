"""User-excluded evaluation of a separate event click-propensity tree."""
import argparse,json,sys,time,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from event_rate_tree import aggregate,fit,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--full',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=np.load(a.features/'raw.npy',mmap_mode='r');fold=np.load(a.features/'fold.npy');names=json.loads((a.features/'names.json').read_text());ev=dict(np.load(a.incumbent))
    y=ev['labels'];base=ev['candidate'];users=ev['users'].reshape(-1,10)[:,0];target=y.reshape(-1,10).sum(1)
    x,offset,feature_names=aggregate(raw,names);start=time.time();records=[]
    if a.full:
        model,spec=fit(x,target,offset,feature_names,{});model.save_model(str(a.out/'tree.txt'));spec['feature_spec']=json.loads((a.features/'spec.json').read_text());spec['training_events']=len(target)
        (a.out/'spec.json').write_text(json.dumps(spec,indent=2));print({'full_events':len(target),'seconds':time.time()-start},flush=True);return
    q=np.empty(len(target))
    for f in range(3):
        tr=fold!=f;te=fold==f;assert not np.intersect1d(users[tr],users[te]).size
        model,spec=fit(x[tr],target[tr],offset[tr],feature_names,{});q[te]=predict(model,x[te],offset[te],spec);model.save_model(str(a.out/f'fold_{f}.txt'))
        record={'fold':f,'train_events':int(tr.sum()),'test_events':int(te.sum()),'users_disjoint':True,'seconds':time.time()-start};records.append(record);print(record,flush=True);del model;gc.collect()
    b=base.reshape(-1,10);pred=(b/b.sum(1,keepdims=True)*q[:,None]).ravel();results=[]
    np.savez_compressed(a.out/'predictions.npz',prediction=pred,event_rate=q,baseline=base,labels=y,indices=ev['indices'])
    for w in [.0625,.125,.25,.5,1.]:
        blend=(1-w)*base+w*pred;gain={};auc={}
        for key,mask in [('combined',np.ones(len(y),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[key]=float(roc_auc_score(y[mask],blend[mask]));gain[key]=auc[key]-float(roc_auc_score(y[mask],base[mask]))
        results.append({'weight':w,'auc':auc,'gain':gain})
    report={'training_spec':spec,'folds':records,'results':results,'base_version':'v18','event_logloss':float(log_loss(target,q)),'baseline_event_logloss':float(log_loss(target,b.sum(1))),
            'limitations':'Reused development cohorts; tree excludes validation users. V18 conditional item probabilities remain unchanged.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
