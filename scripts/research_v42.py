"""Fixed three-seed narrow-memory ensemble against scored V38."""
import argparse,gc,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from v33_training import train
from calibration_model import predict
from rebuild_benchmark import restore,sha,write

WEIGHTS={'base':.095703125,'memory':.095703125,'small':.109375,'slate':.125,'tree':.06280517578125}
MASS=sum(WEIGHTS.values())

def datasets(folder):
    ds=[];align=[]
    for day in ['thursday','friday','weekend']:
        p=folder/'features'/day;al=dict(np.load(p/'alignment.npz'));raw=np.load(p/'raw.npy',mmap_mode='r')
        ds.append({'x':raw,'labels':al['labels'],'names':np.asarray(json.loads((p/'names.json').read_text()))});align.append(al)
    for fold in range(3):
        used=np.unique(np.concatenate([a['user'][a['fold']!=fold] for a in align]));held=np.unique(np.concatenate([a['user'][a['fold']==fold] for a in align]))
        assert not np.intersect1d(used,held).size
    return ds,align

def positions(a):return a.get('first_position',a.get('matched_position'))

def fit(a,plan,ds,align):
    seed=a.seed;assert seed in plan['new_seeds'];out=a.out/f'seed_{seed}'/a.stage;out.mkdir(parents=True,exist_ok=True)
    frozen={'plan_sha256':sha(a.plan),'trainer_sha256':sha(REPO/'src/v33_training.py'),'script_sha256':sha(Path(__file__)),'seed':seed,'stage':a.stage,'feature_sha256':{label:sha(a.benchmark/'features'/label/'raw.npy') for label in ['thursday','friday','weekend']}}
    if (out/'inputs.json').exists():assert json.loads((out/'inputs.json').read_text())==frozen
    write(out/'inputs.json',frozen);records=[];predictions={name:np.full(len(positions(al))*10,np.nan) for name,al in zip(['thursday','friday','weekend'],align)}
    for fold in (range(3) if a.stage=='folds' else [None]):
        keys=[0] if a.stage=='forward' else [0,1,2]
        masks=[align[k]['fold']!=fold if fold is not None else np.ones(len(align[k]['user']),bool) for k in keys]
        name=f'fold_{fold}' if fold is not None else 'model';path=out/(name+'.pt');done=out/(name+'_completed.json')
        if done.exists():
            record=json.loads(done.read_text());assert sha(path)==record['sha256'];model=torch.load(path,map_location='cpu',weights_only=False)
        else:
            if path.exists():raise ValueError('Unverified existing model: '+str(path))
            start=time.time();model=train([ds[k] for k in keys],masks,seed=seed,**plan['training'])
            partial=path.with_suffix('.partial');torch.save(model,partial);partial.rename(path)
            record={'fold':fold,'seed':seed,'sha256':sha(path),'training_events':model['train_events'],'seconds':time.time()-start,'users_disjoint':fold is not None};write(done,record)
        if a.stage=='folds':
            for label,data,al in zip(['thursday','friday','weekend'],ds,align):
                pos=positions(al);test=np.flatnonzero(al['fold'][pos]==fold)
                for start in range(0,len(test),1500):
                    events=test[start:start+1500];rows=(pos[events,None]*10+np.arange(10)).ravel();target=(events[:,None]*10+np.arange(10)).ravel()
                    predictions[label][target]=predict(model,data['x'][rows])
        elif a.stage=='forward':
            pos=positions(align[1]);parts=[]
            for start in range(0,len(pos),1500):
                rows=(pos[start:start+1500,None]*10+np.arange(10)).ravel();parts.append(predict(model,ds[1]['x'][rows]))
            np.savez_compressed(out/'predictions.npz',prediction=np.concatenate(parts),indices=align[1]['indices'][pos])
        records.append(record);print(record,flush=True);del model;gc.collect()
    if a.stage=='folds':
        assert all(np.isfinite(v).all() for v in predictions.values());np.savez_compressed(out/'predictions.npz',**predictions)
    write(out/'report.json',{'inputs':frozen,'fits':records})

def mix(base,parts,average,recipe,mass=1.):
    mode=recipe['mode']
    if mode=='replace_small':return base+WEIGHTS['small']/mass*(average-parts['small'])
    if mode=='replace_pooled':
        keys=['base','memory','small'];weight=sum(WEIGHTS[k] for k in keys)
        return base+(weight*average-sum(WEIGHTS[k]*parts[k] for k in keys))/mass
    w=recipe['weight'];return ((1-w)*mass*base+w*average)/((1-w)*mass+w)

def aggregate(a,plan,inputs):
    ev=dict(np.load(inputs/'evidence/matched_v38.npz'));base=ev['candidate'];y=ev['labels']
    controls={day:dict(np.load(a.benchmark/'predictions'/f'{day}.npz')) for day in ['thursday','friday','weekend']}
    new=[dict(np.load(a.out/f'seed_{s}/folds/predictions.npz')) for s in plan['new_seeds']]
    mean={day:(controls[day]['small']+sum(p[day] for p in new))/3 for day in controls}
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1};ba={k:float(roc_auc_score(y[m],base[m])) for k,m in masks.items()};rows=[];predictions={}
    for recipe in plan['recipes']:
        pred=mix(base,controls['weekend'],mean['weekend'],recipe);assert pred.min()>=0 and pred.reshape(-1,10).sum(1).max()<=1+1e-8
        gains={k:float(roc_auc_score(y[m],pred[m])-ba[k]) for k,m in masks.items()};loss=float(log_loss(y,pred)-log_loss(y,base));weekday={}
        for day in ['thursday','friday']:
            ref=np.load(inputs/f'evidence/{day}_components.npz');cp=mix(ref['candidate'],controls[day],mean[day],recipe,MASS)
            weekday[day]=float(roc_auc_score(ref['labels'],cp)-roc_auc_score(ref['labels'],ref['candidate']))
        eligible=gains['combined']>=1e-5 and gains['earlier']>0 and gains['later']>0 and loss<=0 and all(x>0 for x in weekday.values())
        row={'recipe':recipe,'gains':gains,'logloss_change':loss,'weekday_component_gains':weekday,'eligible':eligible};rows.append(row);predictions[recipe['name']]=pred
    valid=[r for r in rows if r['eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid);order={'replace_small':0,'blend_0125':1,'blend_0250':2,'replace_pooled':3}
        selected=min([r for r in valid if r['gains']['combined']>=best-2e-6],key=lambda r:order[r['recipe']['name']])
        np.savez_compressed(a.out/'selected_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],period=ev['period'],baseline=base,candidate=predictions[selected['recipe']['name']])
    report={'plan':plan,'plan_sha256':sha(a.plan),'baseline_auc':ba,'results':rows,'selected':selected,'official_auc':None,'portal_uploads':0};write(a.out/'report.json',report);print(json.dumps(report),flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--seed',type=int)
    p.add_argument('--stage',choices=['folds','full','forward','aggregate'],required=True);p.add_argument('--plan',type=Path,default=REPO/'results/v42_experiment_plan.json');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());inputs,_=restore(a.benchmark)
    if a.stage=='aggregate':aggregate(a,plan,inputs)
    else:fit(a,plan,*datasets(a.benchmark))

if __name__=='__main__':main()
