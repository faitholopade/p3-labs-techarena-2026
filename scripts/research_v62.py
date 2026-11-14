"""Fixed whole-ensemble transfer of the pre-cutoff user-response residual."""
import argparse
import json
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from utility_transfer import transfer
from rebuild_benchmark import sha, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['benchmark', 'v50', 'v51', 'v61', 'out']: p.add_argument('--'+k, type=Path, required=True)
    p.add_argument('--stage', choices=['screen', 'forward-score'], required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    path = REPO / 'results/v62_experiment_plan.json'; plan = json.loads(path.read_text())
    if a.stage == 'forward-score':
        selected = json.loads((a.out/'report.json').read_text())['selected']; assert selected
        ev = dict(np.load(a.v51/'forward_evidence.npz')); base = ev['candidate']; y = ev['labels']
        new = np.load(a.v61/'forward/predictions.npz'); old = np.load(a.v50/'forward/predictions.npz')
        assert np.array_equal(new['indices'],ev['indices']) and np.array_equal(old['indices'],ev['indices'])
        candidate = transfer(base,new['prediction'],old['prediction'],selected['recipe']['strength'])
        report=dict(recipe=selected['recipe'],baseline_auc=float(roc_auc_score(y,base)),candidate_auc=float(roc_auc_score(y,candidate)),
                    gain=float(roc_auc_score(y,candidate)-roc_auc_score(y,base)),logloss_change=float(log_loss(y,candidate)-log_loss(y,base)),
                    events=len(ev['indices']),not_full_incumbent_validation=True,scope=plan['limits'],official_score=None,portal_uploads=0)
        np.savez_compressed(a.out/'forward_evidence.npz',indices=ev['indices'],users=ev['users'],labels=y,baseline=base,candidate=candidate)
        write(a.out/'forward_report.json',report);print(json.dumps(report));return
    ev = dict(np.load(a.v51/'selected_evidence.npz')); base=ev['candidate'];y=ev['labels']
    current=dict(np.load(a.v51/'current_predictions.npz'));narrow=dict(np.load(a.v51/'narrow_predictions.npz'))
    new=dict(np.load(a.v61/'folds/predictions.npz'));rows=[];candidates={}
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    for recipe in plan['recipes']:
        candidate=transfer(base,new['weekend'],narrow['weekend'],recipe['strength'])
        gains={k:float(roc_auc_score(y[m],candidate[m])-roc_auc_score(y[m],base[m])) for k,m in masks.items()}
        loss=float(log_loss(y,candidate)-log_loss(y,base));weekdays={}
        for day in ['thursday','friday']:
            ref=np.load(a.benchmark/f'inputs/evidence/{day}_components.npz')
            changed=transfer(current[day],new[day],narrow[day],recipe['strength'])
            weekdays[day]=float(roc_auc_score(ref['labels'],changed)-roc_auc_score(ref['labels'],current[day]))
        eligible=gains['combined']>=.000010 and min(gains.values())>0 and loss<=0 and min(weekdays.values())>0
        rows.append(dict(recipe=recipe,gains=gains,logloss_change=loss,weekday_component_gains=weekdays,eligible=eligible));candidates[recipe['name']]=candidate
    valid=[r for r in rows if r['eligible']];selected=None
    if valid:
        best=max(r['gains']['combined'] for r in valid);selected=min([r for r in valid if r['gains']['combined']>=best-.000005],key=lambda r:r['recipe']['strength'])
        np.savez_compressed(a.out/'selected_evidence.npz',**{k:ev[k] for k in ['indices','labels','users','period']},baseline=base,candidate=candidates[selected['recipe']['name']])
    report=dict(plan=plan,plan_sha256=sha(path),source_sha256=sha(Path(__file__)),transfer_source_sha256=sha(REPO/'src/utility_transfer.py'),
                model_plan_sha256=sha(REPO/'results/v61_experiment_plan.json'),baseline_auc=float(roc_auc_score(y,base)),results=rows,selected=selected,official_score=None,portal_uploads=0)
    write(a.out/'report.json',report);print(json.dumps(report))


if __name__=='__main__':main()
