"""Fixed memory-tree experiment against scored V38; no portal operations."""
import argparse,gc,json,sys,time
from pathlib import Path
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
import v39_tree
from screen_v29_pooled import load_datasets
from screen_v30_pooled import position,atomic_npz
from backtest_v24_factor import sha256,write_json

TREE=.06280517578125
WEIGHTS={'base':.095703125,'memory':.095703125,'small':.109375,'slate':.125,'tree':TREE}
MASS=sum(WEIGHTS.values())

def load(a):
    inputs=json.loads(a.inputs.read_text())
    paths={k:a.workspace/v for k,v in inputs['paths'].items()}
    for k,digest in inputs['sha256'].items():
        assert sha256(paths[k])==digest, 'Input mismatch: '+k
    ds,al=load_datasets(SimpleNamespace(weekdays=paths['weekdays'],weekends=paths['weekends']),388)
    return paths,ds,al

def predict_events(model,spec,raw,events):
    parts=[]
    for start in range(0,len(events),2000):
        rows=(events[start:start+2000,None]*10+np.arange(10)).ravel()
        parts.append(v39_tree.predict(model,raw[rows],spec))
    return np.concatenate(parts)

def fit(a,plan):
    paths,ds,al=load(a);config=plan['variants'][a.variant]
    out=a.out/a.variant
    if a.stage!='folds':out=out/a.stage
    out.mkdir(parents=True,exist_ok=True)
    frozen={'plan_sha256':sha256(a.plan),'inputs_sha256':sha256(a.inputs),
            'config':config,'stage':a.stage,
            'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'src/v39_tree.py']}}
    if (out/'plan.json').exists():assert json.loads((out/'plan.json').read_text())==frozen
    write_json(out/'plan.json',frozen)
    if (out/'report.json').exists():
        report=json.loads((out/'report.json').read_text())
        for rec in report['fits']:assert sha256(out/rec['file'])==rec['model_sha256']
        print({'already_completed':str(out)},flush=True);return
    labels=['thursday','friday','weekend'];records=[]
    predictions={name:np.full(10*len(position(align)),np.nan) for name,align in zip(labels,al)}
    loops=range(3) if a.stage=='folds' else [None]
    for fold in loops:
        indices=[0] if a.stage=='forward' else [0,1,2]
        masks=[al[k]['fold']!=fold if fold is not None else np.ones(len(al[k]['user']),bool) for k in indices]
        name=f'fold_{fold}' if fold is not None else 'model'
        modelpath=out/(name+'.txt');specpath=out/(name+'_spec.json');done=out/(name+'_completed.json')
        start=time.time()
        if done.exists():
            record=json.loads(done.read_text());assert sha256(modelpath)==record['model_sha256']
            assert sha256(specpath)==record['spec_sha256']
            model=lgb.Booster(model_file=str(modelpath));spec=json.loads(specpath.read_text())
        else:
            if modelpath.exists():raise ValueError('Unverified checkpoint already exists')
            if fold is not None:
                held=np.unique(np.concatenate([al[k]['user'][~m] for k,m in zip(indices,masks)]))
                train=np.unique(np.concatenate([al[k]['user'][m] for k,m in zip(indices,masks)]))
                assert not np.intersect1d(held,train).size
            model,spec=v39_tree.fit([ds[k] for k in indices],masks,config)
            partial=modelpath.with_suffix('.partial');model.save_model(str(partial));partial.replace(modelpath)
            write_json(specpath,spec)
            record={'fold':fold,'file':modelpath.name,'spec_file':specpath.name,
                    'model_sha256':sha256(modelpath),'spec_sha256':sha256(specpath),
                    'training_events':sum(int(m.sum()) for m in masks),'users_disjoint':fold is not None,
                    'seconds':time.time()-start}
            write_json(done,record)
        if a.stage=='folds':
            for label,data,align in zip(labels,ds,al):
                pos=position(align);at=np.flatnonzero(align['fold'][pos]==fold)
                target=(at[:,None]*10+np.arange(10)).ravel()
                predictions[label][target]=predict_events(model,spec,data['x'],pos[at])
        elif a.stage=='forward':
            pos=position(al[1]);rows=(pos[:,None]*10+np.arange(10)).ravel()
            pred=predict_events(model,spec,ds[1]['x'],pos)
            atomic_npz(out/'predictions.npz',prediction=pred,indices=al[1]['indices'][pos],
                       labels=al[1]['labels'][rows],users=np.repeat(al[1]['user'][pos],10))
        records.append(record);print(record,flush=True);del model;gc.collect()
    if a.stage=='folds':
        assert all(np.isfinite(v).all() for v in predictions.values())
        atomic_npz(out/'predictions.npz',**predictions)
    report={'plan':frozen,'fits':records}
    if (out/'predictions.npz').exists():report['predictions_sha256']=sha256(out/'predictions.npz')
    write_json(out/'report.json',report)

def mix(base,oldtree,new,recipe,mass=1.):
    w=recipe['weight']
    if recipe['mode']=='replace_tree':return base+w/mass*(new-oldtree)
    return ((1-w)*mass*base+w*new)/((1-w)*mass+w)

def aggregate(a,plan):
    paths,ds,al=load(a);ev=dict(np.load(paths['incumbent']))
    assert np.array_equal(ev['indices'],al[2]['indices'][position(al[2])])
    base=ev['candidate'];y=ev['labels'];controls={}
    for name in WEIGHTS:
        controls[name]=dict(np.load(paths[name]))
    memory=dict(np.load(paths['memory_weekdays']));memory['weekend']=controls['memory']['prediction'];controls['memory']=memory
    h={label:sum(w*controls[name][label] for name,w in WEIGHTS.items())/MASS for label in ['thursday','friday','weekend']}
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    auc={n:float(roc_auc_score(y[m],base[m])) for n,m in masks.items()}
    ys={}
    for name,align in zip(['thursday','friday'],al):
        pos=position(align);rows=(pos[:,None]*10+np.arange(10)).ravel();ys[name]=align['labels'][rows]
    results=[];preds={}
    for variant in plan['variants']:
        folder=a.out/variant;report=json.loads((folder/'report.json').read_text())
        assert sha256(folder/'predictions.npz')==report['predictions_sha256']
        new=dict(np.load(folder/'predictions.npz'))
        for recipe in plan['mixtures']:
            pred=mix(base,controls['tree']['weekend'],new['weekend'],recipe)
            assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1
            assert pred.reshape(-1,10).sum(1).max()<=1+1e-8
            gains={n:float(roc_auc_score(y[m],pred[m])-auc[n]) for n,m in masks.items()}
            weekday={name:float(roc_auc_score(ys[name],mix(h[name],controls['tree'][name],new[name],recipe,MASS))-roc_auc_score(ys[name],h[name])) for name in ys}
            loss=float(log_loss(y,pred)-log_loss(y,base));name=f"{variant}_{recipe['mode']}_{recipe['weight']}"
            eligible=gains['combined']>=.00001 and gains['earlier']>0 and gains['later']>0 and loss<=0 and all(v>0 for v in weekday.values())
            row={'name':name,'variant':variant,'recipe':recipe,'gains':gains,'weekday_component_gains':weekday,'logloss_change':loss,'eligible':eligible}
            results.append(row);preds[name]=pred
    eligible=[r for r in results if r['eligible']];selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible)
        selected=min([r for r in eligible if r['gains']['combined']>=best-.000002],key=lambda r:(r['recipe']['mode']!='replace_tree',r['recipe']['weight'],-r['gains']['combined']))
        atomic_npz(a.out/'selected_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],period=ev['period'],baseline=base,candidate=preds[selected['name']])
    report={'plan':plan,'plan_sha256':sha256(a.plan),'inputs_sha256':sha256(a.inputs),'baseline_auc':auc,'results':results,'selected':selected,'official_score':None,'portal_uploads':0}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

def forward_check(a,plan):
    paths,ds,al=load(a);selection=json.loads((a.out/'report.json').read_text())['selected'];assert selection
    folder=a.out/selection['variant']/'forward';pred=dict(np.load(folder/'predictions.npz'))
    original=dict(np.load(paths['forward_tree']));slate=np.load(paths['forward_slate'])['prediction']
    small=dict(np.load(paths['forward_small']));bm=np.load(paths['forward_base_memory'])
    for other in [original,small]:
        for key in ['indices','labels','users']:assert np.array_equal(pred[key],other[key])
    base=(TREE*original['prediction']+.125*slate+.109375*small['new_head']+.095703125*(bm['base_20261029']+bm['memory_20261029']))/MASS
    candidate=mix(base,original['prediction'],pred['prediction'],selection['recipe'],MASS);y=pred['labels']
    report={'variant':selection['variant'],'recipe':selection['recipe'],'baseline_auc':float(roc_auc_score(y,base)),'candidate_auc':float(roc_auc_score(y,candidate)), 'gain':float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),'logloss_change':float(log_loss(y,candidate)-log_loss(y,base)),'scope':plan['forward'],'not_full_ensemble':True}
    atomic_npz(a.out/'forward_evidence.npz',**{k:pred[k] for k in ['indices','labels','users']},baseline=base,candidate=candidate)
    write_json(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['workspace','inputs','plan','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['folds','full','forward','aggregate','forward_check'],required=True)
    p.add_argument('--variant',choices=['memory7','memory15']);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text())
    if a.stage=='aggregate':aggregate(a,plan)
    elif a.stage=='forward_check':forward_check(a,plan)
    else:fit(a,plan)

if __name__=='__main__':main()
