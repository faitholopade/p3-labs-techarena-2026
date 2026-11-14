"""Fixed V25 memory / V26 seed-control experiment on original V20 user folds."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from daily_training import train
from calibration_model import predict

def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','incumbent','old-neural','out','plan']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());raw=np.load(a.features/'raw.npy',mmap_mode='r')
    alignment=dict(np.load(a.features/'alignment.npz'));names=json.loads((a.features/'names.json').read_text())
    ev=dict(np.load(a.incumbent));old=np.load(a.old_neural/'predictions.npz')['prediction']
    pos=alignment['matched_position'];matched_rows=(pos[:,None]*10+np.arange(10)).ravel()
    assert np.array_equal(alignment['indices'][pos],ev['indices'])
    assert np.array_equal(alignment['labels'][matched_rows],ev['labels'])
    assert np.array_equal(np.repeat(alignment['period'][pos],10),ev['period'])
    original=ev['candidate'].astype(np.float64)
    spec={'plan':plan,'feature_report_sha256':digest(a.features/'report.json'),
          'baseline_sha256':digest(a.incumbent),'old_neural_sha256':digest(a.old_neural/'predictions.npz'),
          'source_sha256':{name:digest(REPO/name) for name in ['scripts/screen_v25_memory.py','src/daily_training.py','src/calibration_model.py']}}
    out_plan=a.out/'plan.json'
    if out_plan.exists() and json.loads(out_plan.read_text())!=spec:raise ValueError('Experiment specification changed.')
    out_plan.write_text(json.dumps(spec,indent=2)+'\n')
    groups={};records=[];started=time.time()
    for seed in plan['seeds']:
        for variant in plan['variants']:
            folder=a.out/f'{variant}_{seed}';folder.mkdir(exist_ok=True)
            columns=344 if variant=='base_control' else len(names)
            ds=[{'x':raw[:,:columns],'labels':alignment['labels'],'names':np.array(names[:columns])}]
            prediction=np.full(len(original),np.nan)
            for fold in range(3):
                training=alignment['fold']!=fold
                test=np.flatnonzero(alignment['fold'][pos]==fold)
                assert not np.intersect1d(alignment['user'][training],alignment['user'][pos[test]]).size
                rows=(pos[test,None]*10+np.arange(10)).ravel();rr=(test[:,None]*10+np.arange(10)).ravel()
                modelpath=folder/f'fold_{fold}.pt';done=folder/f'fold_{fold}_completed.json'
                if done.exists():
                    record=json.loads(done.read_text())
                    assert record['model_sha256']==digest(modelpath)
                    model=torch.load(modelpath,map_location='cpu',weights_only=False)
                else:
                    if modelpath.exists():raise ValueError('Unverified partial model; use a new output directory.')
                    model=train(ds,[training],seed=seed,epochs=20,width=32,rank_weight=.5,threads=3,
                                architecture='plain',offset_name='posterior1.0_mix1.0_utility')
                    torch.save(model,modelpath)
                    record={'variant':variant,'seed':seed,'fold':fold,'training_events':int(training.sum()),
                            'validation_events':len(test),'users_disjoint':True,'selected_features':len(model['selected']),
                            'model_sha256':digest(modelpath),'elapsed_seconds':time.time()-started}
                    done.write_text(json.dumps(record,indent=2)+'\n')
                prediction[rr]=predict(model,raw[rows,:columns]);records.append(record)
                print(record,flush=True);del model;gc.collect()
            assert np.isfinite(prediction).all() and np.max(prediction.reshape(-1,10).sum(1))<=1+1e-7
            np.savez_compressed(folder/'predictions.npz',prediction=prediction,indices=ev['indices'])
            groups[(variant,seed)]=prediction
    memory=np.mean([groups[('exposure_memory',s)] for s in plan['seeds']],axis=0)
    control=np.mean([groups[('base_control',s)] for s in plan['seeds']],axis=0)
    seedmean=(old+sum(groups[('base_control',s)] for s in plan['seeds']))/3
    means={'v25':memory,'v26':seedmean};results=[];all_predictions={}
    masks={'combined':np.ones(len(original),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    baseauc={name:float(roc_auc_score(ev['labels'][mask],original[mask])) for name,mask in masks.items()}
    for family,head in means.items():
        for mode,weight in [('blend',.125),('blend',.25),('replace_neural',.125)]:
            pred=(1-weight)*original+weight*head if mode=='blend' else original+weight*(head-old)
            assert np.isfinite(pred).all() and pred.min()>=0 and pred.reshape(-1,10).sum(1).max()<=1+1e-7
            name=f'{family}_{mode}_{weight}'
            gain={key:float(roc_auc_score(ev['labels'][mask],pred[mask]))-baseauc[key] for key,mask in masks.items()}
            result={'name':name,'family':family,'mode':mode,'weight':weight,'gains':gain,
                    'both_dates_improve':gain['earlier']>0 and gain['later']>0,
                    'logloss_change':float(log_loss(ev['labels'],pred)-log_loss(ev['labels'],original))}
            results.append(result);all_predictions[name]=pred
    controls={key:float(roc_auc_score(ev['labels'][mask],memory[mask])-roc_auc_score(ev['labels'][mask],control[mask])) for key,mask in masks.items()}
    positive=[row for row in results if row['both_dates_improve'] and row['gains']['combined']>0]
    selected=None
    if positive:
        best=max(row['gains']['combined'] for row in positive)
        candidates=[row for row in positive if row['gains']['combined']>=best-.000005]
        selected=min(candidates,key=lambda row:(row['mode']!='replace_neural',row['weight'],-row['gains']['combined']))
        np.savez_compressed(a.out/'selected_evidence.npz',indices=ev['indices'],labels=ev['labels'],
                            baseline=original,candidate=all_predictions[selected['name']],period=ev['period'],
                            users=np.repeat(alignment['user'][pos],10))
    np.savez_compressed(a.out/'head_means.npz',memory=memory,control=control,seedmean=seedmean,old=old)
    report={'spec':spec,'fits':records,'results':results,'memory_minus_control_head_auc':controls,
            'selected':selected,'elapsed_seconds':time.time()-started,'portal_uploads':0,
            'limitation':'Reused weekend cohorts; controls share seeds, but changing feature dimensionality changes initial draws. No official gain implied.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
