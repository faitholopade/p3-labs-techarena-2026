"""Cross-fit a fresh smooth calibration head on expanded daily features."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from daily_training import train
from calibration_model import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['features','incumbent','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--epochs',type=int,default=20);p.add_argument('--width',type=int,default=32)
    p.add_argument('--rank-weight',type=float,default=.5);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--seed',type=int,default=20261104);p.add_argument('--full',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.time()
    raw=np.load(a.features/'raw.npy',mmap_mode='r');al=dict(np.load(a.features/'alignment.npz'))
    names=np.array(json.loads((a.features/'names.json').read_text()));ev=dict(np.load(a.incumbent));position=al['matched_position']
    assert np.array_equal(al['matched_indices'],ev['indices'])
    ds=[{'x':raw,'labels':al['labels'],'names':names}]
    config={'epochs':a.epochs,'width':a.width,'rank_weight':a.rank_weight,'seed':a.seed,'threads':a.threads,'offset_name':'posterior1.0_mix1.0_utility'}
    if a.full:
        model=train(ds,[np.ones(len(al['indices']),bool)],**config);torch.save(model,a.out/'model.pt')
        (a.out/'spec.json').write_text(json.dumps({'config':config,'training_events':len(al['indices']),'feature_spec':json.loads((a.features/'spec.json').read_text())},indent=2));return
    prediction=np.empty(len(ev['labels']));records=[]
    for fold in range(3):
        training=al['fold']!=fold;test=np.flatnonzero(al['fold'][position]==fold)
        assert not np.intersect1d(al['user'][training],al['user'][position[test]]).size
        model=train(ds,[training],**config);torch.save(model,a.out/f'fold_{fold}.pt')
        rows=(position[test,None]*10+np.arange(10)).ravel();te=(test[:,None]*10+np.arange(10)).ravel()
        prediction[te]=predict(model,raw[rows]);rec={'fold':fold,'training_events':int(training.sum()),'validation_events':len(test),'users_disjoint':True,'seconds':time.time()-start}
        records.append(rec);print(rec,flush=True);del model;gc.collect()
    np.savez_compressed(a.out/'predictions.npz',prediction=prediction,baseline=ev['candidate'],labels=ev['labels'],indices=ev['indices'])
    results=[]
    for weight in [.0625,.125,.25,.5,1.]:
        candidate=(1-weight)*ev['candidate']+weight*prediction;gain={};auc={}
        for label,mask in [('combined',np.ones(len(candidate),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[label]=float(roc_auc_score(ev['labels'][mask],candidate[mask]));gain[label]=auc[label]-float(roc_auc_score(ev['labels'][mask],ev['candidate'][mask]))
        results.append({'weight':weight,'auc':auc,'gain':gain})
    (a.out/'report.json').write_text(json.dumps({'config':config,'folds':records,'results':results,'base_version':'v19','limits':'Reused development cohorts; calibration fitting excludes held-out users across both dates.'},indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
