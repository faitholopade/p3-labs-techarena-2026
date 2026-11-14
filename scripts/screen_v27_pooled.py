"""Cross-fit a fixed calibration head with two added chronological weekdays."""
import argparse,gc,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from daily_training import train
from calibration_model import predict
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['weekdays','weekends','incumbent','old-neural','control','plan','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--full',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    frozen=json.loads(a.plan.read_text());datasets=[];alignments=[]
    for folder in [a.weekdays/'thursday',a.weekdays/'friday',a.weekends]:
        x=np.load(folder/'raw.npy',mmap_mode='r');al=dict(np.load(folder/'alignment.npz'))
        names=np.array(json.loads((folder/'names.json').read_text())[:344])
        datasets.append({'x':x[:,:344],'labels':al['labels'],'names':names});alignments.append(al)
    assert all(np.array_equal(datasets[0]['names'],ds['names']) for ds in datasets)
    users=np.concatenate([al['user'] for al in alignments]);folds=np.concatenate([al['fold'] for al in alignments])
    lookup=np.full(int(users.max())+1,-1,dtype=np.int8)
    for f in range(3):
        held=np.unique(users[folds==f]);assert np.all(lookup[held]<0);lookup[held]=f
    ev=dict(np.load(a.incumbent));old=np.load(a.old_neural/'predictions.npz')['prediction']
    control=np.load(a.control/'predictions.npz')['prediction']
    base=ev['candidate'].astype(np.float64);al=alignments[-1];position=al['matched_position']
    assert np.array_equal(al['indices'][position],ev['indices'])
    plan={'recipe':frozen,'weekday_reports':{d:sha256(a.weekdays/d/'report.json') for d in ['thursday','friday']},
          'incumbent_sha256':sha256(a.incumbent),'source_sha256':sha256(Path(__file__)),
          'full':a.full,'events_by_cohort':[len(cohort['indices']) for cohort in alignments]}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen plan changed.')
    write_json(path,plan);config=dict(seed=20261029,epochs=20,width=32,rank_weight=.5,threads=3,architecture='plain',offset_name='posterior1.0_mix1.0_utility')
    if a.full:
        if (a.out/'model.pt').exists():raise ValueError('Existing full model.')
        model=train(datasets,[np.ones(len(x['indices']),bool) for x in alignments],**config)
        torch.save(model,a.out/'model.pt');write_json(a.out/'report.json',{'plan':plan,'model_sha256':sha256(a.out/'model.pt')});return
    prediction=np.full(len(base),np.nan);records=[];started=time.time()
    for fold in range(3):
        training=[x['fold']!=fold for x in alignments];test=np.flatnonzero(al['fold'][position]==fold)
        assert not np.intersect1d(np.concatenate([x['user'][m] for x,m in zip(alignments,training)]),al['user'][position[test]]).size
        modelpath=a.out/f'fold_{fold}.pt';done=a.out/f'fold_{fold}_completed.json'
        if done.exists():
            record=json.loads(done.read_text());assert record['model_sha256']==sha256(modelpath)
            model=torch.load(modelpath,map_location='cpu',weights_only=False)
        else:
            if modelpath.exists():raise ValueError('Unverified partial model.')
            model=train(datasets,training,**config);torch.save(model,modelpath)
            record={'fold':fold,'training_events':sum(int(m.sum()) for m in training),'validation_events':len(test),'users_disjoint':True,'model_sha256':sha256(modelpath),'seconds':time.time()-started}
            write_json(done,record)
        rows=(position[test,None]*10+np.arange(10)).ravel();rr=(test[:,None]*10+np.arange(10)).ravel()
        prediction[rr]=predict(model,datasets[-1]['x'][rows]);records.append(record);print(record,flush=True);del model;gc.collect()
    assert np.isfinite(prediction).all()
    np.savez_compressed(a.out/'predictions.npz',indices=ev['indices'],prediction=prediction)
    masks={'combined':np.ones(len(base),bool),'earlier':ev['period']==0,'later':ev['period']==1};rows=[];saved={}
    for mode,weight in [('blend',.125),('blend',.25),('replace_neural',.125)]:
        pred=(1-weight)*base+weight*prediction if mode=='blend' else base+weight*(prediction-old)
        gains={key:float(roc_auc_score(ev['labels'][m],pred[m])-roc_auc_score(ev['labels'][m],base[m])) for key,m in masks.items()}
        name=f'{mode}_{weight}';saved[name]=pred
        rows.append({'name':name,'mode':mode,'weight':weight,'gains':gains,'both_dates_improve':gains['earlier']>0 and gains['later']>0,'logloss_change':float(log_loss(ev['labels'],pred)-log_loss(ev['labels'],base))})
    eligible=[row for row in rows if row['both_dates_improve'] and row['gains']['combined']>0];selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible);near=[r for r in eligible if r['gains']['combined']>=best-.000005]
        selected=min(near,key=lambda r:(r['mode']!='replace_neural',r['weight'],-r['gains']['combined']))
        np.savez_compressed(a.out/'selected_evidence.npz',indices=ev['indices'],labels=ev['labels'],baseline=base,candidate=saved[selected['name']],period=ev['period'],users=np.repeat(al['user'][position],10))
    report={'plan':plan,'folds':records,'results':rows,'selected':selected,
            'head_auc_change_vs_same_seed_weekend_control':{key:float(roc_auc_score(ev['labels'][m],prediction[m])-roc_auc_score(ev['labels'][m],control[m])) for key,m in masks.items()},
            'seconds':time.time()-started,'portal_uploads':0,'limits':'Comparison remains on reused weekend cohorts. Adding weekday training is not itself an independent weekday test.'}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
