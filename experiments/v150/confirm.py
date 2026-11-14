"""Confirm the frozen exploratory V147 full-correlation choice transfer."""
import importlib.util,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v150'
spec=importlib.util.spec_from_file_location('v147_support',ROOT/'experiments/v147/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    report={}
    for split in ['primary','secondary']:
        e=np.load(ROOT/f'runtime/v142/{split}_evidence.npz');idx=e['indices'];base=e['candidate'];y=e['labels'];where={int(x):i for i,x in enumerate(idx)};original=np.zeros((len(idx),11));treated=np.zeros_like(original)
        for cohort in ([0] if split=='primary' else [1,2]):
            p=np.load(ROOT/f'runtime/v147/cohort{cohort}/predictions.npz');j=np.array([where[int(x)] for x in p['indices']]);original[j]=p['0.0'];treated[j]=p['1.0']
        assert (original>0).all() and (treated>0).all();pred=m.m.transfer(base,original,treated,'choice',1.);ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][idx];masks={'all':np.ones(len(y),bool)}
        if split=='primary':masks.update(first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10))
        else:masks.update(saturday=np.repeat(ts<1737849600,10),sunday=np.repeat(ts>=1737849600,10))
        scores={k:{'baseline_auc':float(roc_auc_score(y[z],base[z])),'candidate_auc':float(roc_auc_score(y[z],pred[z])),'gain':float(roc_auc_score(y[z],pred[z])-roc_auc_score(y[z],base[z]))} for k,z in masks.items()};ll=float(log_loss(y,pred)-log_loss(y,base));report[split]=dict(metrics=scores,logloss_change=ll,all_calendar_slices_positive=all(x['gain']>0 for x in scores.values()));np.savez_compressed(OUT/f'{split}_evidence.npz',indices=idx,labels=y,users=e['users'],period=e['period'],baseline=base,candidate=pred);print(split,report[split],flush=True)
    report['secondary_confirmation_passed']=report['secondary']['all_calendar_slices_positive'] and report['secondary']['logloss_change']<=0;report['original_primary_gate_passed']=False;report['scope']='Reused V142 proxies; exact V142 heldout head unavailable.';m.m.write(OUT/'confirmation.json',report)
if __name__=='__main__':main()
