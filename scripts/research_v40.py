"""Fixed candidate-conditioned slate comparison against scored V38."""
import argparse,gc,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from v40_training import train
from v40_model import predict
from research_v39 import load,WEIGHTS,MASS,TREE
from screen_v30_pooled import position,atomic_npz
from backtest_v24_factor import sha256,write_json

def predict_events(model,raw,events):
    parts=[]
    for start in range(0,len(events),2000):
        rows=(events[start:start+2000,None]*10+np.arange(10)).ravel()
        parts.append(predict(model,raw[rows]))
    return np.concatenate(parts)

def fit(a,plan):
    paths,ds,al=load(a);config=plan['variants']['candidate_set32'];out=a.out/'candidate_set32'
    if a.stage!='folds':out=out/a.stage
    out.mkdir(parents=True,exist_ok=True)
    frozen={'plan_sha256':sha256(a.plan),'inputs_sha256':sha256(a.inputs),'config':config,'stage':a.stage,
            'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'src/v40_model.py',REPO/'src/v40_training.py',REPO/'scripts/research_v39.py']}}
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==frozen
    write_json(out/'plan.json',frozen)
    if (out/'report.json').exists():
        report=json.loads((out/'report.json').read_text())
        for rec in report['fits']:assert sha256(out/rec['file'])==rec['model_sha256']
        print({'already_completed':str(out)},flush=True);return
    labels=['thursday','friday','weekend'];records=[]
    predictions={name:np.full(10*len(position(align)),np.nan) for name,align in zip(labels,al)}
    for fold in (range(3) if a.stage=='folds' else [None]):
        keys=[0] if a.stage=='forward' else [0,1,2]
        masks=[al[k]['fold']!=fold if fold is not None else np.ones(len(al[k]['user']),bool) for k in keys]
        name=f'fold_{fold}' if fold is not None else 'model';path=out/(name+'.pt');done=out/(name+'_completed.json');started=time.time()
        if done.exists():
            record=json.loads(done.read_text());assert sha256(path)==record['model_sha256'];model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            if path.exists():raise ValueError('Unverified model already exists')
            if fold is not None:
                held=np.unique(np.concatenate([al[k]['user'][~m] for k,m in zip(keys,masks)]))
                used=np.unique(np.concatenate([al[k]['user'][m] for k,m in zip(keys,masks)]));assert not np.intersect1d(held,used).size
            model=train([ds[k] for k in keys],masks,**config)
            temporary=path.with_suffix('.partial');torch.save(model,temporary);temporary.replace(path)
            record={'fold':fold,'file':path.name,'model_sha256':sha256(path),'training_events':model['train_events'],'users_disjoint':fold is not None,'seconds':time.time()-started}
            write_json(done,record)
        if a.stage=='folds':
            for label,data,align in zip(labels,ds,al):
                pos=position(align);at=np.flatnonzero(align['fold'][pos]==fold);target=(at[:,None]*10+np.arange(10)).ravel()
                predictions[label][target]=predict_events(model,data['x'],pos[at])
        elif a.stage=='forward':
            pos=position(al[1]);rows=(pos[:,None]*10+np.arange(10)).ravel()
            atomic_npz(out/'predictions.npz',prediction=predict_events(model,ds[1]['x'],pos),indices=al[1]['indices'][pos],labels=al[1]['labels'][rows],users=np.repeat(al[1]['user'][pos],10))
        records.append(record);print(record,flush=True);del model;gc.collect()
    if a.stage=='folds':
        assert all(np.isfinite(v).all() for v in predictions.values());atomic_npz(out/'predictions.npz',**predictions)
    report={'plan':frozen,'fits':records}
    if (out/'predictions.npz').exists():report['predictions_sha256']=sha256(out/'predictions.npz')
    write_json(out/'report.json',report)

def mix(base,old_slate,new,recipe,mass=1.):
    w=recipe['weight']
    if recipe['mode']=='replace_slate':return base+w/mass*(new-old_slate)
    assert recipe['mode']=='blend'
    return ((1-w)*mass*base+w*new)/((1-w)*mass+w)

def aggregate(a,plan):
    paths,ds,al=load(a);ev=dict(np.load(paths['incumbent']));base=ev['candidate'];y=ev['labels']
    assert np.array_equal(ev['indices'],al[2]['indices'][position(al[2])])
    controls={name:dict(np.load(paths[name])) for name in WEIGHTS}
    memory=dict(np.load(paths['memory_weekdays']));memory['weekend']=controls['memory']['prediction'];controls['memory']=memory
    h={label:sum(w*controls[name][label] for name,w in WEIGHTS.items())/MASS for label in ['thursday','friday','weekend']}
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1};auc={n:float(roc_auc_score(y[m],base[m])) for n,m in masks.items()}
    ys={}
    for name,align in zip(['thursday','friday'],al):
        pos=position(align);rows=(pos[:,None]*10+np.arange(10)).ravel();ys[name]=align['labels'][rows]
    folder=a.out/'candidate_set32';report=json.loads((folder/'report.json').read_text());assert sha256(folder/'predictions.npz')==report['predictions_sha256']
    new=dict(np.load(folder/'predictions.npz'));results=[];preds={}
    for recipe in plan['mixtures']:
        pred=mix(base,controls['slate']['weekend'],new['weekend'],recipe)
        assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1 and pred.reshape(-1,10).sum(1).max()<=1+1e-8
        gains={n:float(roc_auc_score(y[m],pred[m])-auc[n]) for n,m in masks.items()}
        weekday={name:float(roc_auc_score(ys[name],mix(h[name],controls['slate'][name],new[name],recipe,MASS))-roc_auc_score(ys[name],h[name])) for name in ys}
        loss=float(log_loss(y,pred)-log_loss(y,base));name=f"candidate_set32_{recipe['mode']}_{recipe['weight']}"
        eligible=gains['combined']>=.00001 and gains['earlier']>0 and gains['later']>0 and loss<=0 and all(v>0 for v in weekday.values())
        results.append({'name':name,'variant':'candidate_set32','recipe':recipe,'gains':gains,'weekday_component_gains':weekday,'logloss_change':loss,'eligible':eligible});preds[name]=pred
    eligible=[r for r in results if r['eligible']];selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible)
        selected=min([r for r in eligible if r['gains']['combined']>=best-.000002],key=lambda r:(r['recipe']['mode']!='replace_slate',r['recipe']['weight'],-r['gains']['combined']))
        atomic_npz(a.out/'selected_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],period=ev['period'],baseline=base,candidate=preds[selected['name']])
    report={'plan':plan,'plan_sha256':sha256(a.plan),'inputs_sha256':sha256(a.inputs),'baseline_auc':auc,'results':results,'selected':selected,'official_score':None,'portal_uploads':0}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

def forward_check(a,plan):
    paths,ds,al=load(a);selection=json.loads((a.out/'report.json').read_text())['selected'];assert selection
    pred=dict(np.load(a.out/'candidate_set32/forward/predictions.npz'));original=dict(np.load(paths['forward_tree']));slate=dict(np.load(paths['forward_slate']));small=dict(np.load(paths['forward_small']));bm=np.load(paths['forward_base_memory'])
    for other in [original,slate,small]:
        for key in ['indices','labels','users']:assert np.array_equal(pred[key],other[key])
    base=(TREE*original['prediction']+.125*slate['prediction']+.109375*small['new_head']+.095703125*(bm['base_20261029']+bm['memory_20261029']))/MASS
    candidate=mix(base,slate['prediction'],pred['prediction'],selection['recipe'],MASS);y=pred['labels']
    report={'variant':'candidate_set32','recipe':selection['recipe'],'baseline_auc':float(roc_auc_score(y,base)),'candidate_auc':float(roc_auc_score(y,candidate)),'gain':float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),'logloss_change':float(log_loss(y,candidate)-log_loss(y,base)),'scope':plan['forward'],'not_full_ensemble':True}
    atomic_npz(a.out/'forward_evidence.npz',**{k:pred[k] for k in ['indices','labels','users']},baseline=base,candidate=candidate)
    write_json(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['workspace','inputs','plan','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['folds','full','forward','aggregate','forward_check'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text())
    if a.stage=='aggregate':aggregate(a,plan)
    elif a.stage=='forward_check':forward_check(a,plan)
    else:fit(a,plan)

if __name__=='__main__':main()
