"""Fixed joint transfer of existing narrow and new wide first-query heads."""
import argparse
import json
import sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from research_v43 import MASS
from rebuild_benchmark import sha,write

WEIGHTS={'narrow':.095703125,'plain':.109375,'slate':.125}


def delta(old,new,recipe):return sum(WEIGHTS[k]*recipe[k]*(new[k]-old[k]) for k in WEIGHTS)


def screen(a,plan):
    ev=dict(np.load(a.v63/'selected_evidence.npz'));current=dict(np.load(a.v63/'current_predictions.npz'))
    assert np.array_equal(ev['candidate'],current['weekend'])
    base=ev['candidate'];y=ev['labels'];old={k:dict(np.load(a.v45/k/'folds/predictions.npz')) for k in ['plain','slate']}
    new={k:dict(np.load(a.v66/k/'folds/predictions.npz')) for k in ['plain','slate']}
    old['narrow']=dict(np.load(a.v50/'folds/predictions.npz'));new['narrow']=dict(np.load(a.v63/'folds/predictions.npz'))
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1};rows=[];candidates={}
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
                         total_changed_weight=sum(WEIGHTS[k]*recipe[k] for k in WEIGHTS),new_models=2,eligible=eligible));candidates[recipe['name']]=candidate
    valid=[r for r in rows if r['eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid)
        selected=min([r for r in valid if r['gains']['combined']>=best-.000005],key=lambda r:r['total_changed_weight'])
        np.savez_compressed(a.out/'selected_evidence.npz',**{k:ev[k] for k in ['indices','users','labels','period']},baseline=base,candidate=candidates[selected['recipe']['name']])
    report=dict(plan=plan,plan_sha256=sha(a.plan),source_sha256=sha(Path(__file__)),model_plan_sha256=sha(REPO/'results/v66_experiment_plan.json'),baseline_auc=float(roc_auc_score(y,base)),results=rows,selected=selected,official_score=None,portal_uploads=0)
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)


def forward_score(a,plan):
    selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
    ev=dict(np.load(a.v63/'forward_evidence.npz'));base=ev['candidate'];y=ev['labels'];old={};new={}
    for family in ['plain','slate']:
        prior=np.load(a.v45/family/'forward/predictions.npz');updated=np.load(a.v66/family/'forward/predictions.npz')
        assert np.array_equal(prior['indices'],ev['indices']) and np.array_equal(updated['indices'],ev['indices'])
        old[family]=prior['prediction'];new[family]=updated['prediction']
    prior=np.load(a.v50/'forward/predictions.npz');updated=np.load(a.v63/'forward/predictions.npz')
    assert np.array_equal(prior['indices'],ev['indices']) and np.array_equal(updated['indices'],ev['indices'])
    old['narrow']=prior['prediction'];new['narrow']=updated['prediction']
    candidate=base+delta(old,new,selected['recipe'])/MASS
    np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],users=ev['users'],labels=y,baseline=base,candidate=candidate)
    report=dict(recipe=selected['recipe'],baseline_auc=float(roc_auc_score(y,base)),candidate_auc=float(roc_auc_score(y,candidate)),gain=float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),logloss_change=float(log_loss(y,candidate)-log_loss(y,base)),events=len(ev['indices']),scope='All three adapted heads start from all Tuesday-Thursday parent fits and adapt on 105180 first daily queries; tested on 35040 first-Friday events. Normalized changed components omit unchanged historical V17/V20.',not_full_incumbent_validation=True,official_score=None,portal_uploads=0)
    write(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['benchmark','v45','v50','v63','v66','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--stage',choices=['screen','forward-score'],required=True)
    p.add_argument('--plan',type=Path,default=REPO/'results/v67_experiment_plan.json');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan=json.loads(a.plan.read_text());assert plan['branch_weights']==WEIGHTS
    if a.stage=='screen':screen(a,plan)
    else:forward_score(a,plan)


if __name__=='__main__':main()
