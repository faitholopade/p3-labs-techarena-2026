"""Fit a daily expanded tree, validating only on the original held-out users/events."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from flexible_tree import fit,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['features','incumbent','v17-evidence','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--rounds',type=int,default=700);p.add_argument('--leaves',type=int,default=7)
    p.add_argument('--min-leaf',type=int,default=2000);p.add_argument('--l2',type=float,default=100.)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--seed',type=int,default=20261103)
    p.add_argument('--full',action='store_true')
    p.add_argument('--resume',action='store_true',help='Reuse complete fold model/spec pairs; never reuse partial output.')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.time()
    raw=np.load(a.features/'raw.npy',mmap_mode='r');alignment=dict(np.load(a.features/'alignment.npz'))
    labels=alignment['labels'];fold=alignment['fold'];users=alignment['user'];position=alignment['matched_position']
    names=json.loads((a.features/'names.json').read_text());ev=dict(np.load(a.incumbent));parent=np.load(a.v17_evidence)['candidate']
    assert np.array_equal(alignment['matched_indices'],ev['indices'])
    matched_rows=(position[:,None]*10+np.arange(10)).ravel();assert np.array_equal(labels[matched_rows],ev['labels'])
    config={'mode':'event','rounds':a.rounds,'leaves':a.leaves,'seed':a.seed,'threads':a.threads,'min_leaf':a.min_leaf,'l2':a.l2}
    if a.full:
        model,spec=fit(raw,labels,names,config);model.save_model(str(a.out/'tree.txt'))
        spec.update({'feature_spec':json.loads((a.features/'spec.json').read_text()),'training_rows':len(labels)})
        (a.out/'spec.json').write_text(json.dumps(spec,indent=2));print({'full_rows':len(labels),'seconds':time.time()-start},flush=True);return
    prediction=np.empty(len(ev['labels']));records=[]
    for f in range(3):
        train=np.flatnonzero(fold!=f);test=np.flatnonzero(fold[position]==f)
        assert not np.intersect1d(users[train],users[position[test]]).size
        tr=(train[:,None]*10+np.arange(10)).ravel();te=(test[:,None]*10+np.arange(10)).ravel()
        model_path=a.out/f'fold_{f}.txt';spec_path=a.out/f'fold_{f}_spec.json';reused=False
        if a.resume and model_path.exists() and spec_path.exists():
            spec=json.loads(spec_path.read_text())
            for key,value in config.items():
                if key!='threads' and spec['config'].get(key)!=value:raise ValueError('Checkpoint configuration mismatch.')
            assert spec['names']==names
            model=lgb.Booster(model_file=str(model_path));assert model.current_iteration()==config['rounds'];reused=True
        else:
            model,spec=fit(raw[tr],labels[tr],names,config);model.save_model(str(model_path));spec_path.write_text(json.dumps(spec,indent=2))
        prediction[te]=predict(model,raw[matched_rows[te]],spec)
        rec={'fold':f,'training_events':len(train),'validation_events':len(test),'users_disjoint':True,'reused_checkpoint':reused,'training_threads':spec['config']['threads'],'seconds':time.time()-start}
        records.append(rec);print(rec,flush=True);del model;gc.collect()
    np.savez_compressed(a.out/'predictions.npz',prediction=prediction,baseline=ev['candidate'],labels=ev['labels'],indices=ev['indices'])
    results=[]
    for weight in [.0625,.125,.25,.5,1.]:
        candidate=(1-weight)*parent+weight*prediction;gains={};aucs={}
        for label,mask in [('combined',np.ones(len(candidate),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            aucs[label]=float(roc_auc_score(ev['labels'][mask],candidate[mask]))
            gains[label]=aucs[label]-float(roc_auc_score(ev['labels'][mask],ev['candidate'][mask]))
        results.append({'new_tree_weight':weight,'weights':[1-weight,weight],'auc':aucs,'gain':gains})
    report={'config':config,'training_spec':spec,'folds':records,'results':results,'base_version':'v19',
            'limits':'Reused development cohorts; calibration fitting excludes held-out users across both dates.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
