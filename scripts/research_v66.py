"""First-query adaptation of the saved V45 plain32/slate32 heads."""
import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from first_query_adaptation import fit as adapt
from calibration_model import predict
from research_v45 import datasets,LABELS
from research_v43 import positions,MASS
from rebuild_benchmark import sha,write

WEIGHTS={'plain':.109375,'slate':.125}


def base_path(a,family,stage,stem):
    if stage=='full':
        spec=json.loads((REPO/'artifacts/v45/spec.json').read_text())
        entry=next(m for m in spec['new_heads'] if m['name']==family+'_20261029')
        path=REPO/entry['path'];expected=entry['sha256']
    else:
        entry=next(m for m in json.loads((REPO/'artifacts/v45/research/index.json').read_text())['models'] if m.get('restore_to')==f'{family}/{stage}/{stem}.pt')
        path=a.v45/family/stage/(stem+'.pt');expected=entry['sha256']
    assert sha(path)==expected
    return path


def fit(a,plan):
    ds,align,hashes=datasets(SimpleNamespace(out=a.v45,benchmark=a.benchmark))
    first=[]
    for al in align:
        m=np.zeros(len(al['user']),bool);m[positions(al)]=True;first.append(m)
    hashes['first_query_training_events']=[int(m.sum()) for m in first]
    assert hashes['first_query_training_events']==[35156,35112,34912,35040,70368]
    sources={name:sha(REPO/name) for name in ['scripts/research_v66.py','src/first_query_adaptation.py','src/calibration_model.py','scripts/research_v45.py','scripts/research_v43.py']}
    for family in (plan['families'] if a.family=='all' else [a.family]):
        output=a.out/family/a.stage;output.mkdir(parents=True,exist_ok=True)
        predictions={day:np.full(len(positions(al))*10,np.nan) for day,al in zip(LABELS[2:],align[2:])}
        for fold in (range(3) if a.stage=='folds' else [None]):
            stem=f'fold_{fold}' if fold is not None else 'model'
            parent=base_path(a,family,a.stage,stem)
            base=torch.load(parent,map_location='cpu',weights_only=False)
            assert base['width']==32 and base['architecture']==('plain' if family=='plain' else 'set')
            inputs=dict(plan_sha256=sha(a.plan),features=hashes,stage=a.stage,family=family,fold=fold,
                        base_model_sha256=sha(parent),sources=sources)
            input_path=output/(stem+'_inputs.json')
            if input_path.exists():assert json.loads(input_path.read_text())==inputs
            write(input_path,inputs)
            model_path=output/(stem+'.pt');done=output/(stem+'_completed.json')
            if done.exists():
                record=json.loads(done.read_text());assert sha(model_path)==record['sha256']
                model=torch.load(model_path,map_location='cpu',weights_only=False)
            else:
                assert not model_path.exists()
                keys=[0,1,2] if a.stage=='forward' else list(range(5))
                masks=[first[k]&(align[k]['fold']!=fold) if fold is not None else first[k] for k in keys]
                start=time.time()
                model=adapt(base,[ds[k] for k in keys],masks,seed=plan['seed'],epochs=plan['training']['epochs'],
                            threads=plan['training']['threads'],rank_weight=plan['training']['rank_weight'],
                            checkpoint_path=output/(stem+'_resume.pt'))
                for key in ['mean','median','std']:assert np.array_equal(model[key],base[key])
                temporary=model_path.with_suffix('.partial');torch.save(model,temporary);temporary.rename(model_path)
                record=dict(family=family,fold=fold,sha256=sha(model_path),training_events=model['train_events'],
                            seconds=time.time()-start,users_disjoint=fold is not None,initial_normalization_preserved=True)
                write(done,record)
            if a.stage=='folds':
                for day,data,al in zip(LABELS[2:],ds[2:],align[2:]):
                    pos=positions(al);test=np.flatnonzero(al['fold'][pos]==fold)
                    for start in range(0,len(test),1500):
                        events=test[start:start+1500];rows=(pos[events,None]*10+np.arange(10)).ravel();dest=(events[:,None]*10+np.arange(10)).ravel()
                        predictions[day][dest]=predict(model,data['x'][rows])
            elif a.stage=='forward':
                pos=positions(align[3]);parts=[]
                for start in range(0,len(pos),1500):
                    rows=(pos[start:start+1500,None]*10+np.arange(10)).ravel();parts.append(predict(model,ds[3]['x'][rows]))
                np.savez_compressed(output/'predictions.npz',prediction=np.concatenate(parts),indices=align[3]['indices'][pos])
            write(output/(stem+'_report.json'),dict(inputs=inputs,fit=record));print(json.dumps(record),flush=True)
            del model,base;gc.collect()
        if a.stage=='folds':
            assert all(np.isfinite(p).all() for p in predictions.values())
            np.savez_compressed(output/'predictions.npz',**predictions)


def delta(old,new,recipe):
    return sum(WEIGHTS[k]*recipe[k]*(new[k]-old[k]) for k in WEIGHTS)


def screen(a,plan):
    ev=dict(np.load(a.v63/'selected_evidence.npz'));current=dict(np.load(a.v63/'current_predictions.npz'))
    assert np.array_equal(ev['candidate'],current['weekend'])
    base=ev['candidate'];y=ev['labels'];old={k:dict(np.load(a.v45/k/'folds/predictions.npz')) for k in WEIGHTS}
    new={k:dict(np.load(a.out/k/'folds/predictions.npz')) for k in WEIGHTS}
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    rows=[];candidates={}
    for recipe in plan['recipes']:
        candidate=base+delta({k:v['weekend'] for k,v in old.items()},{k:v['weekend'] for k,v in new.items()},recipe)
        assert np.isfinite(candidate).all() and candidate.min()>0 and candidate.reshape(-1,10).sum(1).max()<=1+1e-8
        gains={k:float(roc_auc_score(y[m],candidate[m])-roc_auc_score(y[m],base[m])) for k,m in masks.items()}
        loss=float(log_loss(y,candidate)-log_loss(y,base));weekday={}
        for day in ['thursday','friday']:
            ref=np.load(a.benchmark/f'inputs/evidence/{day}_components.npz')
            cp=current[day]+delta({k:v[day] for k,v in old.items()},{k:v[day] for k,v in new.items()},recipe)/MASS
            weekday[day]=float(roc_auc_score(ref['labels'],cp)-roc_auc_score(ref['labels'],current[day]))
        eligible=gains['combined']>=.000010 and min(gains.values())>0 and loss<=0 and min(weekday.values())>0
        rows.append(dict(recipe=recipe,gains=gains,logloss_change=loss,weekday_component_gains=weekday,
                         total_changed_weight=sum(WEIGHTS[k]*recipe[k] for k in WEIGHTS),eligible=eligible))
        candidates[recipe['name']]=candidate
    valid=[r for r in rows if r['eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid)
        selected=min([r for r in valid if r['gains']['combined']>=best-.000005],key=lambda r:r['total_changed_weight'])
        np.savez_compressed(a.out/'selected_evidence.npz',**{k:ev[k] for k in ['indices','users','labels','period']},baseline=base,candidate=candidates[selected['recipe']['name']])
    report=dict(plan=plan,plan_sha256=sha(a.plan),source_sha256=sha(Path(__file__)),baseline_auc=float(roc_auc_score(y,base)),results=rows,selected=selected,official_score=None,portal_uploads=0)
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)


def forward_score(a,plan):
    selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
    ev=dict(np.load(a.v63/'forward_evidence.npz'));base=ev['candidate'];y=ev['labels'];old={};new={}
    for family in WEIGHTS:
        prior=np.load(a.v45/family/'forward/predictions.npz');assert np.array_equal(prior['indices'],ev['indices']);old[family]=prior['prediction']
        if selected['recipe'][family]:
            updated=np.load(a.out/family/'forward/predictions.npz');assert np.array_equal(updated['indices'],ev['indices']);new[family]=updated['prediction']
        else:new[family]=old[family]
    candidate=base+delta(old,new,selected['recipe'])/MASS
    np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],users=ev['users'],labels=y,baseline=base,candidate=candidate)
    report=dict(recipe=selected['recipe'],baseline_auc=float(roc_auc_score(y,base)),candidate_auc=float(roc_auc_score(y,candidate)),
                gain=float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),logloss_change=float(log_loss(y,candidate)-log_loss(y,base)),events=len(ev['indices']),
                scope='Selected V66 heads start from all Tuesday-Thursday parent fits and adapt on 105180 first daily queries; tested on 35040 first-Friday events. Normalized changed components omit unchanged historical V17/V20.',not_full_incumbent_validation=True,official_score=None,portal_uploads=0)
    write(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['benchmark','v45','v63','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=['folds','forward','full','screen','forward-score'],required=True)
    p.add_argument('--family',choices=['all','plain','slate'],default='all')
    p.add_argument('--plan',type=Path,default=REPO/'results/v66_experiment_plan.json');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());assert plan['branch_weights']==WEIGHTS
    if a.stage=='screen':screen(a,plan)
    elif a.stage=='forward-score':forward_score(a,plan)
    else:fit(a,plan)


if __name__=='__main__':main()
