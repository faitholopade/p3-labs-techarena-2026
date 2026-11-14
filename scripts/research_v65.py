"""Fixed six-day event-mass correction versus the scored V63 incumbent."""
import argparse,json,sys,time
from pathlib import Path
from types import SimpleNamespace
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from bounded_history import base_logits
from event_mass_correction import features,fit as train,delta,transfer
from rebuild_benchmark import sha,write
from research_v45 import datasets,LABELS
from research_v43 import positions


def fit(a,plan):
    torch.set_num_threads(2)
    ds,align,hashes=datasets(SimpleNamespace(out=a.v45,benchmark=a.benchmark))
    fold=a.fold if a.stage=='folds' else None
    assert a.stage!='folds' or fold in [0,1,2]
    stem=f'fold_{fold}' if fold is not None else 'model';base_path=a.v50/a.stage/(stem+'.pt')
    expected={m['restore_to']:m['sha256'] for m in json.loads((REPO/'artifacts/v50/research/index.json').read_text())['models']}
    assert sha(base_path)==expected[f'{a.stage}/{stem}.pt']
    base=torch.load(base_path,map_location='cpu',weights_only=False)
    for data in ds:
        data['utility']=base_logits(base,data['x'])
        data['event_features'],data['event_names']=features(data['x'],data['names'].tolist(),data['utility'])
    folder=a.out/a.stage;folder.mkdir(parents=True,exist_ok=True)
    frozen=dict(plan_sha256=sha(a.plan),features=hashes,base_model_sha256=sha(base_path),stage=a.stage,fold=fold,
                sources={p:sha(REPO/p) for p in ['scripts/research_v65.py','src/event_mass_correction.py','src/bounded_history.py','src/calibration_model.py','scripts/research_v45.py']})
    inp=folder/(stem+'_inputs.json')
    if inp.exists():assert json.loads(inp.read_text())==frozen
    write(inp,frozen);path=folder/(stem+'.pt');done=folder/(stem+'_completed.json')
    if done.exists():
        record=json.loads(done.read_text());assert sha(path)==record['sha256'];model=torch.load(path,map_location='cpu',weights_only=False)
    else:
        assert not path.exists(),'Unverified model file'
        keys=[0,1,2] if a.stage=='forward' else list(range(5))
        masks=[align[k]['fold']!=fold if fold is not None else np.ones(len(align[k]['user']),bool) for k in keys]
        started=time.time();model=train(base,[ds[k] for k in keys],masks,seed=plan['seed'],epochs=plan['training']['epochs'],threads=plan['training']['threads'],checkpoint_path=folder/(stem+'_resume.pt'))
        assert all(torch.equal(v,base['state'][k]) for k,v in model['base_spec']['state'].items())
        temp=path.with_suffix('.partial');torch.save(model,temp);temp.replace(path)
        record=dict(fold=fold,sha256=sha(path),training_events=model['train_events'],seconds=time.time()-started,users_disjoint=fold is not None,base_frozen=True)
        write(done,record)
    if a.stage=='folds':
        predictions={}
        for name,data,al in zip(LABELS[2:],ds[2:],align[2:]):
            pos=positions(al);test=np.flatnonzero(al['fold'][pos]==fold)
            predictions[name]=np.full(len(pos),np.nan);predictions[name][test]=delta(model,data['event_features'][pos[test]])
        np.savez_compressed(folder/(stem+'_delta.npz'),**predictions)
    elif a.stage=='forward':
        pos=positions(align[3]);np.savez_compressed(folder/'delta.npz',delta=delta(model,ds[3]['event_features'][pos]),indices=align[3]['indices'][pos])
    write(folder/(stem+'_report.json'),dict(inputs=frozen,fit=record));print(json.dumps(record),flush=True)


def screen(a,plan):
    evidence=dict(np.load(a.v63/'selected_evidence.npz'));baseline=evidence['candidate'];y=evidence['labels']
    current=dict(np.load(a.v63/'current_predictions.npz'));assert np.array_equal(baseline,current['weekend'])
    partials=[dict(np.load(a.out/f'folds/fold_{k}_delta.npz')) for k in range(3)];deltas={}
    for day in ['weekend','thursday','friday']:
        values=np.stack([p[day] for p in partials]);assert np.all(np.isfinite(values).sum(0)==1)
        deltas[day]=np.nansum(values,axis=0)
    np.savez_compressed(a.out/'folds/delta.npz',**deltas)
    masks={'combined':np.ones(len(y),bool),'earlier':evidence['period']==0,'later':evidence['period']==1};rows=[];candidates={}
    for recipe in plan['recipes']:
        candidate=transfer(baseline,deltas['weekend'],recipe['strength'])
        gains={k:float(roc_auc_score(y[m],candidate[m])-roc_auc_score(y[m],baseline[m])) for k,m in masks.items()}
        loss=float(log_loss(y,candidate)-log_loss(y,baseline));weekdays={}
        for day in ['thursday','friday']:
            ref=np.load(a.benchmark/f'inputs/evidence/{day}_components.npz');changed=transfer(current[day],deltas[day],recipe['strength'])
            weekdays[day]=float(roc_auc_score(ref['labels'],changed)-roc_auc_score(ref['labels'],current[day]))
        eligible=gains['combined']>=.000010 and min(gains.values())>0 and loss<=0 and min(weekdays.values())>0
        row=dict(recipe=recipe,gains=gains,logloss_change=loss,weekday_component_gains=weekdays,eligible=eligible);rows.append(row);candidates[recipe['name']]=candidate
    valid=[r for r in rows if r['eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid);selected=min([r for r in valid if r['gains']['combined']>=best-.000005],key=lambda r:r['recipe']['strength'])
        np.savez_compressed(a.out/'selected_evidence.npz',**{k:evidence[k] for k in ['indices','labels','users','period']},baseline=baseline,candidate=candidates[selected['recipe']['name']])
    report=dict(plan=plan,plan_sha256=sha(a.plan),source_sha256=sha(Path(__file__)),baseline_auc=float(roc_auc_score(y,baseline)),results=rows,selected=selected,official_score=None,portal_uploads=0)
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)


def forward(a,plan):
    selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
    ev=dict(np.load(a.v63/'forward_evidence.npz'));d=np.load(a.out/'forward/delta.npz');assert np.array_equal(d['indices'],ev['indices'])
    base=ev['candidate'];y=ev['labels'];candidate=transfer(base,d['delta'],selected['recipe']['strength'])
    report=dict(recipe=selected['recipe'],baseline_auc=float(roc_auc_score(y,base)),candidate_auc=float(roc_auc_score(y,candidate)),gain=float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),logloss_change=float(log_loss(y,candidate)-log_loss(y,base)),events=len(ev['indices']),not_full_incumbent_validation=True,scope='Preselected event-only transfer to normalized V63 changed components. Correction and V50 base fitted on Tuesday-Thursday only; no conditional choice change. Historical V17/V20 omitted.',official_score=None,portal_uploads=0)
    np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],baseline=base,candidate=candidate)
    write(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['benchmark','v45','v50','v63','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',choices=['folds','forward','full','screen','forward-score'],required=True)
    p.add_argument('--fold',type=int,choices=[0,1,2]);p.add_argument('--plan',type=Path,default=REPO/'results/v65_experiment_plan.json')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text())
    if a.stage=='screen':screen(a,plan)
    elif a.stage=='forward-score':forward(a,plan)
    else:fit(a,plan)


if __name__=='__main__':main()
