"""Pooled context/capacity/tree comparisons with explicit weekday head diagnostics."""
import argparse,gc,json,sys,time,zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score,log_loss
from daily_training import train
from calibration_model import predict as neural_predict
from v30_features import FeatureView,CONTEXT_NAMES
import v30_tree
from backtest_v24_factor import sha256,write_json

def atomic_npz(path,**data):
    tmp=path.with_suffix('.partial.npz');np.savez_compressed(tmp,**data)
    with zipfile.ZipFile(tmp) as z:
        if z.testzip() is not None:raise ValueError('Corrupt saved prediction archive.')
    tmp.replace(path)

def datasets(a,context):
    ctx=np.load(a.train_data)['context'];ds=[];align=[]
    for label,folder in [('thursday',a.weekdays/'thursday'),('friday',a.weekdays/'friday'),('weekend',a.weekends)]:
        al=dict(np.load(folder/'alignment.npz'));raw=np.load(folder/'raw.npy',mmap_mode='r');names=json.loads((folder/'names.json').read_text())[:344]
        c=ctx[al['indices']];x=FeatureView(raw,c if context else None)
        if context:names=names+CONTEXT_NAMES
        ds.append({'name':label,'x':x,'raw_all':raw,'labels':al['labels'],'names':np.asarray(names),'folder':folder});align.append(al)
    assert all(np.array_equal(ds[0]['names'],d['names']) for d in ds)
    users=np.concatenate([al['user'] for al in align]);folds=np.concatenate([al['fold'] for al in align]);seen=set()
    for f in range(3):
        held=set(users[folds==f].tolist());assert not held.intersection(seen);seen.update(held)
    return ds,align

def position(al):return al.get('matched_position',al.get('first_position'))

def predict_events(model,spec,kind,x,events):
    result=[]
    for start in range(0,len(events),2000):
        rows=(events[start:start+2000,None]*10+np.arange(10)).ravel();raw=x[rows]
        result.append(neural_predict(model,raw) if kind=='neural' else v30_tree.predict(model,raw,spec))
    return np.concatenate(result)

def controls(a,plan):
    ds,align=datasets(a,False);out=a.out/'controls';out.mkdir(parents=True,exist_ok=True)
    existing=np.load(a.old_head/'predictions.npz');ev=np.load(a.incumbent);assert np.array_equal(existing['indices'],ev['indices'])
    result={'weekend':existing['prediction']};memory={};audit=[]
    for k in range(2):
        al=align[k];pos=position(al);pred=np.full(10*len(pos),np.nan);mem=pred.copy()
        for fold in range(3):
            at=np.flatnonzero(al['fold'][pos]==fold);rows=(at[:,None]*10+np.arange(10)).ravel()
            p=a.old_head/f'fold_{fold}.pt';done=json.loads((a.old_head/f'fold_{fold}_completed.json').read_text());assert sha256(p)==done['model_sha256']
            model=torch.load(p,map_location='cpu',weights_only=False);assert model['names']==ds[k]['names'].tolist()
            pred[rows]=predict_events(model,None,'neural',ds[k]['x'],pos[at]);del model
            p=a.memory_head/f'fold_{fold}.pt';done=json.loads((a.memory_head/f'fold_{fold}_completed.json').read_text());assert sha256(p)==done['model_sha256']
            model=torch.load(p,map_location='cpu',weights_only=False);mem[rows]=predict_events(model,None,'neural',ds[k]['raw_all'],pos[at]);del model;gc.collect()
        assert np.isfinite(pred).all() and np.isfinite(mem).all();result[ds[k]['name']]=pred;memory[ds[k]['name']]=mem
        rows=(pos[:,None]*10+np.arange(10)).ravel();y=al['labels'][rows]
        audit.append({'cohort':ds[k]['name'],'rows':len(y),'v27_head_auc':float(roc_auc_score(y,pred)),
                      'v29_head_auc':float(roc_auc_score(y,mem)),'v29_head_gain':float(roc_auc_score(y,mem)-roc_auc_score(y,pred)),
                      'half_v29_half_v27_head_gain':float(roc_auc_score(y,.5*mem+.5*pred)-roc_auc_score(y,pred))})
    atomic_npz(out/'v27_predictions.npz',**result);atomic_npz(out/'v29_weekday_predictions.npz',**memory)
    write_json(out/'report.json',{'plan':plan,'v29_weekday_diagnostic':audit,'limits':'User-excluded component-head comparisons on existing weekdays, not full V27 or V29 ensemble scores; these dates have been used in prior research.'})
    print(json.dumps(audit),flush=True)

def fit_model(ds,masks,config):
    if config['kind']=='tree':return v30_tree.fit(ds,masks,config)
    model=train(ds,masks,seed=config['seed'],epochs=20,width=config['width'],rank_weight=.5,threads=2,
                architecture='plain',offset_name='posterior1.0_mix1.0_utility')
    return model,None

def save_model(out,name,model,spec,kind):
    path=out/(name+('.pt' if kind=='neural' else '.txt'));tmp=path.with_suffix('.partial')
    if kind=='neural':torch.save(model,tmp)
    else:model.save_model(str(tmp));write_json(out/(name+'_spec.json'),spec)
    tmp.replace(path);return path

def run(a,plan):
    config=plan['variants'][a.variant];ds,align=datasets(a,config['context']);out=a.out/a.variant
    if a.full:out=out/'full'
    if a.forward:out=out/'forward'
    out.mkdir(parents=True,exist_ok=True)
    record={'configuration':config,'plan_sha256':sha256(a.plan),'incumbent_sha256':sha256(a.incumbent),'full':a.full,'forward':a.forward,
            'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'src/v30_features.py',REPO/'src/v30_tree.py',REPO/'src/daily_training.py']},
            'feature_report_sha256':[sha256(d['folder']/'report.json') for d in ds],'train_data_sha256':sha256(a.train_data)}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text())!=record:raise ValueError('Frozen run changed.')
    write_json(out/'plan.json',record)
    if a.full or a.forward:
        if (out/'report.json').exists():raise ValueError('Completed full/forward fit already exists.')
        indices=range(1) if a.forward else range(3);masks=[np.ones(len(align[k]['indices']),bool) for k in indices]
        model,spec=fit_model([ds[k] for k in indices],masks,config);path=save_model(out,'model',model,spec,config['kind'])
        final={'plan':record,'model_sha256':sha256(path),'training_events':sum(int(m.sum()) for m in masks)}
        if a.forward:
            pos=position(align[1]);prediction=predict_events(model,spec,config['kind'],ds[1]['x'],pos)
            control=torch.load(a.forward_control,map_location='cpu',weights_only=False)
            assert control['seed']==20261029 and control['train_events']==len(align[0]['indices'])
            baseline=predict_events(control,None,'neural',FeatureView(ds[1]['raw_all']),pos);rows=(pos[:,None]*10+np.arange(10)).ravel();y=align[1]['labels'][rows]
            half=.5*baseline+.5*prediction
            final['diagnostic']={'control_auc':float(roc_auc_score(y,baseline)),'new_head_auc':float(roc_auc_score(y,prediction)),
                                 'half_blend_gain':float(roc_auc_score(y,half)-roc_auc_score(y,baseline)),
                                 'logloss_change':float(log_loss(y,half)-log_loss(y,baseline)),
                                 'scope':'Train Thursday only, test first Friday events. Standalone head diagnostic; not a complete V27 comparison.'}
            atomic_npz(out/'evidence.npz',indices=align[1]['indices'][pos],labels=y,baseline=baseline,candidate=half,head=prediction,users=np.repeat(align[1]['user'][pos],10))
        write_json(out/'report.json',final);print(json.dumps(final),flush=True);return
    predictions={d['name']:np.full(10*len(position(al)),np.nan) for d,al in zip(ds,align)};fold_records=[]
    for fold in range(3):
        masks=[al['fold']!=fold for al in align];held=np.unique(np.concatenate([al['user'][~m] for al,m in zip(align,masks)]));train_users=np.unique(np.concatenate([al['user'][m] for al,m in zip(align,masks)]));assert not np.intersect1d(held,train_users).size
        path=out/(f'fold_{fold}'+('.pt' if config['kind']=='neural' else '.txt'));done=out/f'fold_{fold}_completed.json';start=time.time()
        if done.exists():
            rec=json.loads(done.read_text());assert sha256(path)==rec['model_sha256']
            model=torch.load(path,map_location='cpu',weights_only=False) if config['kind']=='neural' else lgb.Booster(model_file=str(path))
            spec=None if config['kind']=='neural' else json.loads((out/f'fold_{fold}_spec.json').read_text())
        else:
            if path.exists():raise ValueError('Unverified checkpoint exists.')
            model,spec=fit_model(ds,masks,config);path=save_model(out,f'fold_{fold}',model,spec,config['kind'])
            rec={'fold':fold,'training_events':sum(int(m.sum()) for m in masks),'users_disjoint':True,'model_sha256':sha256(path),'seconds':time.time()-start};write_json(done,rec)
        for d,al in zip(ds,align):
            pos=position(al);at=np.flatnonzero(al['fold'][pos]==fold);rows=(at[:,None]*10+np.arange(10)).ravel()
            predictions[d['name']][rows]=predict_events(model,spec,config['kind'],d['x'],pos[at])
        fold_records.append(rec);print(rec,flush=True);del model;gc.collect()
    assert all(np.isfinite(x).all() for x in predictions.values());atomic_npz(out/'predictions.npz',**predictions)
    write_json(out/'report.json',{'plan':record,'folds':fold_records})

def aggregate(a,plan):
    ds,align=datasets(a,False);ev=dict(np.load(a.incumbent));base=ev['candidate'];old=np.load(a.out/'controls/v27_predictions.npz');oldtree=np.load(a.old_tree)['prediction']
    assert np.array_equal(ev['indices'],align[-1]['indices'][position(align[-1])]);masks={'combined':np.ones(len(base),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    baseline={n:float(roc_auc_score(ev['labels'][m],base[m])) for n,m in masks.items()};allrows=[];saved={};checks={}
    for name,config in plan['variants'].items():
        pred=np.load(a.out/name/'predictions.npz');weekday={}
        for k in range(2):
            cohort=ds[k]['name'];pos=position(align[k]);rows=(pos[:,None]*10+np.arange(10)).ravel();y=align[k]['labels'][rows];mix=.5*old[cohort]+.5*pred[cohort]
            weekday[cohort]={'gain':float(roc_auc_score(y,mix)-roc_auc_score(y,old[cohort])),
                              'head_gain':float(roc_auc_score(y,pred[cohort])-roc_auc_score(y,old[cohort]))}
        checks[name]=weekday
        for mixture in plan['mixtures'][config['kind']]:
            mode=mixture['mode'];w=mixture['weight']
            if mode=='replace_pooled_head':p=base+w*(pred['weekend']-old['weekend'])
            elif mode=='replace_old_tree':p=base+w*(pred['weekend']-oldtree)
            else:p=(1-w)*base+w*pred['weekend']
            assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1 and p.reshape(-1,10).sum(1).max()<=1+1e-7
            gains={n:float(roc_auc_score(ev['labels'][m],p[m])-baseline[n]) for n,m in masks.items()};key=f'{name}_{mode}_{w}';saved[key]=p
            row={'name':key,'variant':name,'mode':mode,'weight':w,'gains':gains,'both_weekend_dates_improve':all(gains[k]>0 for k in ['earlier','later']),
                 'both_weekday_head_checks_improve':all(x['gain']>0 for x in weekday.values()),'logloss_change':float(log_loss(ev['labels'],p)-log_loss(ev['labels'],base))}
            allrows.append(row)
    eligible=[r for r in allrows if r['both_weekend_dates_improve'] and r['both_weekday_head_checks_improve'] and r['gains']['combined']>=plan['minimum_exploratory_gain'] and r['logloss_change']<=0]
    selected=None
    if eligible:
        top=max(r['gains']['combined'] for r in eligible);near=[r for r in eligible if r['gains']['combined']>=top-plan['tie_tolerance']]
        selected=min(near,key=lambda r:(r['mode']=='blend',r['weight'],-r['gains']['combined']))
        atomic_npz(a.out/'selected_evidence.npz',indices=ev['indices'],labels=ev['labels'],baseline=base,candidate=saved[selected['name']],period=ev['period'],users=ev['users'])
    report={'plan':plan,'incumbent':'v27','official_incumbent_auc':.832593,'baseline_auc':baseline,'results':allrows,'weekday_head_checks':checks,'selected':selected,
            'limits':'Weekend scores compare full candidates with V27. Weekday checks compare a fixed equal-weight head mixture with the V27 head alone, not the full ensemble. All cohorts have prior research exposure.','portal_uploads':0}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['weekdays','weekends','train-data','incumbent','old-head','memory-head','old-tree','plan','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--variant');p.add_argument('--controls',action='store_true');p.add_argument('--aggregate',action='store_true');p.add_argument('--full',action='store_true');p.add_argument('--forward',action='store_true');p.add_argument('--forward-control',type=Path)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text())
    if a.controls:controls(a,plan)
    elif a.aggregate:aggregate(a,plan)
    else:run(a,plan)

if __name__=='__main__':main()
