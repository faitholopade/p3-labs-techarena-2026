"""Frozen V29 comparison: pooled seeds, history memory, balanced sampling, CE."""
import argparse,gc,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from daily_training import train
from calibration_model import predict
from backtest_v24_factor import sha256,write_json

def load_datasets(a,columns):
    datasets=[];alignments=[]
    for folder in [a.weekdays/'thursday',a.weekdays/'friday',a.weekends]:
        raw=np.load(folder/'raw.npy',mmap_mode='r');al=dict(np.load(folder/'alignment.npz'))
        names=np.asarray(json.loads((folder/'names.json').read_text())[:columns])
        datasets.append({'x':raw[:,:columns],'labels':al['labels'],'names':names});alignments.append(al)
    assert all(np.array_equal(datasets[0]['names'],d['names']) for d in datasets)
    users=np.concatenate([al['user'] for al in alignments]);folds=np.concatenate([al['fold'] for al in alignments])
    seen=set()
    for f in range(3):
        u=set(users[folds==f].tolist());assert not seen.intersection(u);seen.update(u)
    return datasets,alignments

def balanced_weights(alignments,exponent):
    """Per user-day counts use cohort identities only, never click labels."""
    result=[]
    for k,al in enumerate(alignments):
        # Weekend cache contains two complete dates; count them separately.
        day=al['period'] if k==2 else np.zeros(len(al['user']),np.int8)
        key=al['user'].astype(np.int64)*2+day
        _,inverse,count=np.unique(key,return_inverse=True,return_counts=True)
        result.append(count[inverse].astype(np.float64)**(-exponent))
    return result

def fit_variant(a,plan):
    config=plan['variants'][a.variant];ds,align=load_datasets(a,config['columns']);out=a.out/a.variant
    if a.full:out=out/'full'
    out.mkdir(parents=True,exist_ok=True)
    inputs={'plan_sha256':sha256(a.plan),'source_sha256':sha256(Path(__file__)),
            'trainer_sha256':sha256(REPO/'src/daily_training.py'),
            'incumbent_sha256':sha256(a.incumbent),'config':config,'full':a.full,
            'feature_reports':{str(p):sha256(p/'report.json') for p in [a.weekdays/'thursday',a.weekdays/'friday',a.weekends]}}
    path=out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=inputs:raise ValueError('Frozen fit changed.')
    write_json(path,inputs)
    kw=dict(seed=config['seed'],epochs=20,width=32,rank_weight=config['rank_weight'],threads=a.threads,
            architecture='plain',offset_name='posterior1.0_mix1.0_utility')
    if config.get('sampling_exponent'):
        kw['sampling_weights']=balanced_weights(align,config['sampling_exponent'])
    ev=dict(np.load(a.incumbent));position=align[-1]['matched_position']
    assert np.array_equal(align[-1]['indices'][position],ev['indices'])
    if a.full:
        if (out/'model.pt').exists():raise ValueError('Existing full model.')
        model=train(ds,[np.ones(len(al['user']),bool) for al in align],**kw)
        torch.save(model,out/'model.pt');write_json(out/'report.json',{'plan':inputs,'model_sha256':sha256(out/'model.pt'),'training_events':model['train_events']});return
    prediction=np.full(len(ev['labels']),np.nan);records=[]
    for fold in range(3):
        masks=[al['fold']!=fold for al in align];test=np.flatnonzero(align[-1]['fold'][position]==fold)
        assert not np.intersect1d(np.concatenate([al['user'][m] for al,m in zip(align,masks)]),align[-1]['user'][position[test]]).size
        modelpath=out/f'fold_{fold}.pt';done=out/f'fold_{fold}_completed.json';started=time.time()
        if done.exists():
            record=json.loads(done.read_text());assert record['model_sha256']==sha256(modelpath)
            model=torch.load(modelpath,map_location='cpu',weights_only=False)
        else:
            if modelpath.exists():raise ValueError('Unverified partial model.')
            model=train(ds,masks,**kw);torch.save(model,modelpath)
            record={'fold':fold,'training_events':sum(int(m.sum()) for m in masks),'validation_events':len(test),
                    'users_disjoint':True,'model_sha256':sha256(modelpath),'seconds':time.time()-started}
            write_json(done,record)
        rows=(position[test,None]*10+np.arange(10)).ravel();target=(test[:,None]*10+np.arange(10)).ravel()
        prediction[target]=predict(model,ds[-1]['x'][rows]);records.append(record);print(record,flush=True)
        del model;gc.collect()
    assert np.isfinite(prediction).all()
    np.savez_compressed(out/'predictions.npz',indices=ev['indices'],prediction=prediction)
    write_json(out/'report.json',{'plan':inputs,'folds':records,'head_auc':float(roc_auc_score(ev['labels'],prediction))})

def aggregate(a,plan):
    ev=dict(np.load(a.incumbent));base=ev['candidate'];old=np.load(a.old_head)['prediction']
    assert len(base)==len(old) and np.allclose(base,.75*ev['baseline']+.25*old,atol=1e-12,rtol=0)
    masks={'combined':np.ones(len(base),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    baseauc={name:roc_auc_score(ev['labels'][m],base[m]) for name,m in masks.items()}
    rows=[];predictions={}
    for name,config in plan['variants'].items():
        raw=np.load(a.out/name/'predictions.npz');assert np.array_equal(raw['indices'],ev['indices'])
        head=raw['prediction'];head=(head+old)/2 if config['head_average_with_v27'] else head
        for recipe in plan['mixtures']:
            mode=recipe['mode'];weight=recipe['weight']
            candidate=base+weight*(head-old) if mode=='replace_pooled_head' else (1-weight)*base+weight*head
            assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1
            assert candidate.reshape(-1,10).sum(1).max()<=1+1e-8
            key=f'{name}_{mode}_{weight}';predictions[key]=candidate
            gains={n:float(roc_auc_score(ev['labels'][m],candidate[m])-baseauc[n]) for n,m in masks.items()}
            rows.append({'name':key,'variant':name,'mode':mode,'weight':weight,'gains':gains,
                         'both_dates_improve':gains['earlier']>0 and gains['later']>0,
                         'logloss_change':float(log_loss(ev['labels'],candidate)-log_loss(ev['labels'],base)),
                         'head_auc':float(roc_auc_score(ev['labels'],head))})
    eligible=[r for r in rows if r['both_dates_improve'] and r['gains']['combined']>=plan['minimum_exploratory_gain'] and r['logloss_change']<=0]
    selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible)
        near=[r for r in eligible if r['gains']['combined']>=best-plan['tie_tolerance']]
        selected=min(near,key=lambda r:(r['mode']!='replace_pooled_head',r['weight'],-r['gains']['combined']))
        np.savez_compressed(a.out/'selected_evidence.npz',indices=ev['indices'],labels=ev['labels'],baseline=base,
                            candidate=predictions[selected['name']],period=ev['period'],users=ev['users'])
    report={'plan':plan,'incumbent':'v27','official_incumbent_auc':.832593,'baseline_auc':baseauc,
            'results':rows,'selected':selected,'portal_uploads':0,
            'limits':'Twelve fixed comparisons on repeatedly reused user-excluded weekend folds. Selection uncertainty is not captured by the conditional bootstrap.'}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['weekdays','weekends','incumbent','old-head','plan','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant');p.add_argument('--full',action='store_true');p.add_argument('--aggregate',action='store_true');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();plan=json.loads(a.plan.read_text());a.out.mkdir(parents=True,exist_ok=True)
    if a.aggregate:aggregate(a,plan)
    else:fit_variant(a,plan)

if __name__=='__main__':main()
