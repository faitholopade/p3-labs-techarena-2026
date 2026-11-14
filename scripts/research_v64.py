"""Fixed remaining first-query-head transfers against scored V63."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path[:0]=[str(REPO/'scripts')]
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from rebuild_benchmark import sha,write
from research_v43 import MASS

def change(new,narrow,plain,r):return .095703125*r['narrow']*(new-narrow)+.109375*r['plain']*(new-plain)

def main():
 p=argparse.ArgumentParser()
 for k in ['benchmark','v45','v50','v63','out']:p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--stage',choices=['screen','forward'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 path=REPO/'results/v64_experiment_plan.json';plan=json.loads(path.read_text())
 if a.stage=='forward':
  selected=json.loads((a.out/'report.json').read_text())['selected'];assert selected
  ev=dict(np.load(a.v63/'forward_evidence.npz'));base=ev['candidate'];y=ev['labels']
  parts=[np.load(f/'forward/predictions.npz') for f in [a.v63,a.v50,a.v45/'plain']]
  assert all(np.array_equal(v['indices'],ev['indices']) for v in parts)
  candidate=base+change(*[v['prediction'] for v in parts],selected['recipe'])/MASS
  report=dict(recipe=selected['recipe'],gain=float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),logloss_change=float(log_loss(y,candidate)-log_loss(y,base)),events=len(ev['indices']),scope='First-Friday normalized changed components, preselected recipe; new head starts with all Tue-Thu base and adapts first queries; unchanged V17/V20 omitted.',official_score=None,portal_uploads=0)
  np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],users=ev['users'],labels=y,baseline=base,candidate=candidate);write(a.out/'forward_report.json',report);print(json.dumps(report));return
 ev=dict(np.load(a.v63/'selected_evidence.npz'));base=ev['candidate'];y=ev['labels'];assert abs(roc_auc_score(y,base)-.8378351482608294)<1e-12
 current=dict(np.load(a.v63/'current_predictions.npz'));new=dict(np.load(a.v63/'folds/predictions.npz'));narrow=dict(np.load(a.v50/'folds/predictions.npz'));plain=dict(np.load(a.v45/'plain/folds/predictions.npz'))
 masks=dict(combined=np.ones(len(y),bool),earlier=ev['period']==0,later=ev['period']==1);rows=[];predictions={}
 for r in plan['recipes']:
  candidate=base+change(new['weekend'],narrow['weekend'],plain['weekend'],r)
  gains={k:float(roc_auc_score(y[m],candidate[m])-roc_auc_score(y[m],base[m])) for k,m in masks.items()};loss=float(log_loss(y,candidate)-log_loss(y,base));week={}
  for day in ['thursday','friday']:
   labels=np.load(a.benchmark/f'inputs/evidence/{day}_components.npz')['labels'];changed=current[day]+change(new[day],narrow[day],plain[day],r)/MASS
   week[day]=float(roc_auc_score(labels,changed)-roc_auc_score(labels,current[day]))
  assert candidate.min()>0 and candidate.reshape(-1,10).sum(1).max()<=1+1e-8
  rows.append(dict(recipe=r,gains=gains,logloss_change=loss,weekday_component_gains=week,total_changed_weight=.095703125*r['narrow']+.109375*r['plain'],eligible=gains['combined']>=1e-5 and min(gains.values())>0 and loss<=0 and min(week.values())>0));predictions[r['name']]=candidate
 valid=[r for r in rows if r['eligible']];selected=None
 if valid:
  best=max(r['gains']['combined'] for r in valid);selected=min([r for r in valid if r['gains']['combined']>=best-5e-6],key=lambda r:r['total_changed_weight'])
  np.savez_compressed(a.out/'selected_evidence.npz',**{k:ev[k] for k in ['indices','labels','users','period']},baseline=base,candidate=predictions[selected['recipe']['name']])
 report=dict(plan=plan,plan_sha256=sha(path),source_sha256=sha(Path(__file__)),baseline_auc=float(roc_auc_score(y,base)),results=rows,selected=selected,official_score=None,portal_uploads=0);write(a.out/'report.json',report);print(json.dumps(report))
if __name__=='__main__':main()
