"""Frozen post-V65 exploratory transfers and slate event-rate residuals."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from bounded_history import base_logits
from event_mass_correction import features as old_features,delta as old_delta,transfer
from slate_event_residual import features,fit,delta
from rebuild_benchmark import sha,write
from research_v43 import positions,MASS
DAYS=['thursday','friday','weekend']

def evaluation(baseline,candidate,labels,period=None):
    masks={'combined':np.ones(len(labels),bool)}
    if period is not None:masks.update(earlier=period==0,later=period==1)
    assert np.isfinite(candidate).all() and candidate.min()>0 and candidate.reshape(-1,10).sum(1).max()<1
    return dict(gains={k:float(roc_auc_score(labels[m],candidate[m])-roc_auc_score(labels[m],baseline[m])) for k,m in masks.items()},logloss_change=float(log_loss(labels,candidate)-log_loss(labels,baseline)))

def first_mask(al):
    mask=np.zeros(len(al['user']),bool)
    if 'period' in al:
        for period in np.unique(al['period']):
            ix=np.flatnonzero(al['period']==period);_,p=np.unique(al['user'][ix],return_index=True);mask[ix[p]]=True
    else:mask[al['first_position']]=True
    return mask

def screen(a,version,deltas=None):
    evidence=dict(np.load(a.work/'v65/selected_evidence.npz'));base=dict(np.load(a.work/'v65/current_predictions.npz'));labels=evidence['labels'];rows=[];predictions={}
    assert abs(roc_auc_score(labels,base['weekend'])-.8378525735530878)<1e-12
    if version==74:
        old=dict(np.load(a.work/'v63/current_predictions.npz'));narrow=dict(np.load(a.work/'v51/narrow_predictions.npz'));adapted=dict(np.load(a.work/'v63/adapted_predictions.npz'));ed=dict(np.load(a.work/'v65/current_event_delta.npz'));recipes=[dict(name='remaining_narrow_full',weight=.095703125)]
    else:recipes=json.loads((ROOT/'results/v74_v75_experiment_plan.json').read_text())['v75']['recipes']
    for recipe in recipes:
        changed={}
        for day in DAYS:
            if version==74:changed[day]=transfer(old[day]+recipe['weight']*(adapted[day]-narrow[day])/(1 if day=='weekend' else MASS),ed[day],1.)
            else:changed[day]=transfer(base[day],deltas[day],recipe['strength'])
        result=evaluation(base['weekend'],changed['weekend'],labels,evidence['period']);week={}
        for day in DAYS[:2]:
            yy=np.load(a.work/f'benchmark/inputs/evidence/{day}_components.npz')['labels'];week[day]=evaluation(base[day],changed[day],yy)['gains']['combined']
        result.update(recipe=recipe,weekday_component_gains=week)
        result['exploratory_eligible']=min(result['gains'].values())>0 and min(week.values())>0 and result['logloss_change']<=0
        result['old_screen_minimum_pass']=result['gains']['combined']>=1e-5
        rows.append(result);predictions[recipe['name']]=changed
    valid=[r for r in rows if r['exploratory_eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid);selected=min([r for r in valid if r['gains']['combined']>=best-5e-6],key=lambda r:r['recipe'].get('strength',1.))
        chosen=predictions[selected['recipe']['name']];np.savez_compressed(a.out/'selected_evidence.npz',**{k:evidence[k] for k in ['indices','labels','users','period']},baseline=base['weekend'],candidate=chosen['weekend']);np.savez_compressed(a.out/'current_predictions.npz',**chosen)
    report=dict(version=f'v{version}',baseline='v65',baseline_auc=float(roc_auc_score(labels,base['weekend'])),plan_sha256=sha(ROOT/'results/v74_v75_experiment_plan.json'),results=rows,selected=selected,official_score=None,portal_uploads=0)
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)

def dataset(a,day,base_model):
    folder=a.work/f'benchmark/features/{day}';al=dict(np.load(folder/'alignment.npz'));raw=np.load(folder/'raw.npy',mmap_mode='r')
    utility=base_logits(base_model['base_spec'],raw);ef,names=old_features(raw,base_model['names'],utility);assert names==base_model['event_names'];ed=old_delta(base_model,ef)
    x,offset=features(utility,ed);return x,offset,al

def folds(a):
    pred={};records=[]
    for fold in range(3):
        path=ROOT/f'artifacts/v65/research/folds/fold_{fold}.pt';base_model=torch.load(path,map_location='cpu',weights_only=False);data=[dataset(a,day,base_model) for day in DAYS]
        masks=[first_mask(al)&(al['fold']!=fold) for _,_,al in data]
        train_users=np.unique(np.concatenate([al['user'][m] for (_,_,al),m in zip(data,masks)]));held=np.unique(np.concatenate([al['user'][al['fold']==fold] for _,_,al in data]));assert not np.intersect1d(train_users,held).size
        xx=np.concatenate([x[m] for (x,_,_),m in zip(data,masks)]);oo=np.concatenate([o[m] for (_,o,_),m in zip(data,masks)]);yy=np.concatenate([al['labels'].reshape(-1,10)[m].sum(1) for (_,_,al),m in zip(data,masks)])
        model=fit(xx,oo,yy);write(a.out/f'fold_{fold}.json',model)
        for day,(x,_,al) in zip(DAYS,data):
            pos=positions(al);test=al['fold'][pos]==fold
            if day not in pred:pred[day]=np.full(len(pos),np.nan)
            assert np.isnan(pred[day][test]).all();pred[day][test]=delta(model,x[pos[test]])
        record=dict(fold=fold,training_events=len(yy),held_users=len(held),users_disjoint=True,base_model_sha256=sha(path),model_sha256=sha(a.out/f'fold_{fold}.json'));records.append(record);write(a.out/'fits.json',records);print(json.dumps(record),flush=True)
    assert all(np.isfinite(v).all() for v in pred.values());np.savez_compressed(a.out/'deltas.npz',**pred);screen(a,75,pred)

def forward(a,version):
    selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
    ev=dict(np.load(a.work/'forward_v65/evidence.npz'));parts=dict(np.load(a.work/'forward_v65/components.npz'))
    if version==74:candidate=transfer(parts['v63']+.095703125*(parts['adapted']-parts['narrow'])/MASS,parts['event_delta'],1.)
    else:
        path=ROOT/'artifacts/v65/research/forward/model.pt';base_model=torch.load(path,map_location='cpu',weights_only=False)
        x,o,al=dataset(a,'thursday',base_model);mask=first_mask(al);model=fit(x[mask],o[mask],al['labels'].reshape(-1,10)[mask].sum(1));write(a.out/'forward.json',model)
        x,_,al=dataset(a,'friday',base_model);pos=positions(al);assert np.array_equal(ev['indices'],al['indices'][pos]);candidate=transfer(ev['candidate'],delta(model,x[pos]),selected['recipe']['strength'])
    report=evaluation(ev['candidate'],candidate,ev['labels']);report.update(events=len(ev['indices']),recipe=selected['recipe'],scope='Chronological first-Friday normalized changed components; historical V17/V20 omitted.',official_score=None)
    np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],users=ev['users'],labels=ev['labels'],baseline=ev['candidate'],candidate=candidate);write(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)

def full(a):
    selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
    forward_report=json.loads((a.out/'forward_report.json').read_text());assert forward_report['gains']['combined']>0 and forward_report['logloss_change']<=0
    base_model=torch.load(ROOT/'artifacts/v65/research/full/model.pt',map_location='cpu',weights_only=False);data=[dataset(a,day,base_model) for day in DAYS];masks=[first_mask(al) for _,_,al in data]
    x=np.concatenate([x[m] for (x,_,_),m in zip(data,masks)]);o=np.concatenate([o[m] for (_,o,_),m in zip(data,masks)]);y=np.concatenate([al['labels'].reshape(-1,10)[m].sum(1) for (_,_,al),m in zip(data,masks)])
    write(a.out/'full.json',fit(x,o,y))

def main():
    p=argparse.ArgumentParser();p.add_argument('--work',type=Path,default=ROOT/'work');p.add_argument('--version',type=int,choices=[74,75],required=True);p.add_argument('--stage',choices=['screen','folds','forward','full'],required=True);a=p.parse_args();a.out=a.work/f'v{a.version}';a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    if a.stage=='screen':screen(a,a.version,dict(np.load(a.out/'deltas.npz')) if a.version==75 else None)
    elif a.stage=='folds':assert a.version==75;folds(a)
    elif a.stage=='forward':forward(a,a.version)
    else:assert a.version==75;full(a)
if __name__=='__main__':main()
