"""Fit conservative residual trees above frozen outer-fold calibration heads."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from flexible_tree import fit,predict
from calibration_residual import features,mixture,CALIBRATION_WEIGHT


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['features','offsets','v20-evidence','v21-evidence','v21-head','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--mode',choices=['event','binary'],default='event');p.add_argument('--rounds',type=int,default=150)
    p.add_argument('--leaves',type=int,default=3);p.add_argument('--min-leaf',type=int,default=3000);p.add_argument('--l2',type=float,default=400)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--full',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);raw=np.load(a.features/'raw.npy',mmap_mode='r');names=json.loads((a.features/'names.json').read_text());al=dict(np.load(a.features/'alignment.npz'));started=time.time()
    config={'mode':a.mode,'rounds':a.rounds,'leaves':a.leaves,'min_leaf':a.min_leaf,'l2':a.l2,'threads':a.threads,'lr':.025,'seed':20261106,'offset_name':'calibration_mix_'+('utility' if a.mode=='event' else 'logit')}
    prediction=np.empty(len(raw));records=[]
    for fold in ['full'] if a.full else [0,1,2]:
        key='full' if fold=='full' else f'fold_{fold}';offset_path=a.offsets/(key+'.npz');meta=json.loads(offset_path.with_suffix('.json').read_text())
        assert hashlib.sha256(offset_path.read_bytes()).hexdigest()==meta['sha256'];off=dict(np.load(offset_path));x,fnames=features(raw,names,off['tree'],off['neural'])
        mask=np.ones(len(al['indices']),bool) if fold=='full' else al['fold']!=fold;train=np.flatnonzero(mask);test=np.flatnonzero(~mask)
        assert not np.intersect1d(al['user'][train],al['user'][test]).size
        rows=(train[:,None]*10+np.arange(10)).ravel();mp=a.out/(key+'.txt');sp=a.out/(key+'_spec.json')
        if a.resume and mp.exists() and sp.exists():
            spec=json.loads(sp.read_text());assert spec['config']==config and spec['names']==fnames;model=lgb.Booster(model_file=str(mp));assert model.current_iteration()==config['rounds']
        else:
            model,spec=fit(x[rows],al['labels'][rows],fnames,config);model.save_model(str(mp));sp.write_text(json.dumps(spec,indent=2))
        if fold!='full':
            rows=(test[:,None]*10+np.arange(10)).ravel();prediction[rows]=predict(model,x[rows],spec)
        rec={'fold':fold,'training_events':len(train),'validation_events':len(test),'users_disjoint':True,'seconds':time.time()-started};records.append(rec);print(rec,flush=True)
        del x,off,model;gc.collect()
    if a.full:
        (a.out/'spec.json').write_text(json.dumps({'config':config,'training_spec':spec,'training_events':len(al['indices']),'offset_spec':json.loads((a.offsets/'spec.json').read_text())},indent=2));return
    v20=dict(np.load(a.v20_evidence));v21=dict(np.load(a.v21_evidence));old=dict(np.load(a.features/'baseline_heads.npz'));new21=np.load(a.v21_head)['prediction']
    r=(al['matched_position'][:,None]*10+np.arange(10)).ravel();assert np.array_equal(v20['indices'],v21['indices']);assert np.array_equal(al['labels'][r],v20['labels'])
    q=mixture(old['tree'],old['neural']);q21=q+(4/15)*(new21-old['neural']);long=np.repeat(al['long_gap'],10);results=[]
    for parent,base,small,weights in [('v20',v20['candidate'],q,[CALIBRATION_WEIGHT/4,CALIBRATION_WEIGHT/2,CALIBRATION_WEIGHT]),('v21',v21['candidate'],q21,[CALIBRATION_WEIGHT/8,CALIBRATION_WEIGHT/4,CALIBRATION_WEIGHT/2])]:
        for weight in weights:
            candidate=base+weight*(prediction[r]-q[r]);auc={};gain={};gain21={}
            for label,mask in [('combined',np.ones(len(candidate),bool)),('earlier',v20['period']==0),('later',v20['period']==1)]:
                auc[label]=float(roc_auc_score(v20['labels'][mask],candidate[mask]));gain[label]=auc[label]-float(roc_auc_score(v20['labels'][mask],v20['candidate'][mask]));gain21[label]=auc[label]-float(roc_auc_score(v21['labels'][mask],v21['candidate'][mask]))
            small_new=small+(weight/CALIBRATION_WEIGHT)*(prediction-q)
            gap=float(roc_auc_score(al['labels'][long],small_new[long])-roc_auc_score(al['labels'][long],small[long]))
            results.append({'parent':parent,'replacement_weight':weight,'auc':auc,'gain_vs_v20':gain,'gain_vs_v21':gain21,'long_gap_calibration_gain_vs_parent':gap})
    np.savez_compressed(a.out/'predictions.npz',prediction=prediction,matched_prediction=prediction[r],labels=al['labels'],indices=al['indices'],cutoffs=al['cutoffs'])
    report={'config':config,'folds':records,'results':results,'limits':'Repeated development reuse. Training offsets are in-sample frozen-head predictions; held-out users are excluded from both model stages. Long-gap diagnostic reuses outcomes and only compares calibration subensembles.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
