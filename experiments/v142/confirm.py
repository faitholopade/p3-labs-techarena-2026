"""Confirm one frozen exploratory weekend correction; no recipe search."""
import json,sys,importlib.util
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v142'
spec=importlib.util.spec_from_file_location('v141_weekend',ROOT/'experiments/v141/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    results={}
    for split in ['primary','secondary']:
        e=np.load(ROOT/f'runtime/v136/{split}_evidence.npz');base=e['candidate'];idx=e['indices'];labels=e['labels'];delta=np.full((len(idx),10),np.nan)
        cohorts=[0] if split=='primary' else [1,2]
        positions={int(x):j for j,x in enumerate(idx)}
        for cohort in cohorts:
            v=np.load(ROOT/f'runtime/v141/cohort{cohort}/predictions.npz');p=np.array([positions[int(x)] for x in v['indices']]);delta[p]=v['shared']
        assert np.isfinite(delta).all();candidate=m.transfer(base,delta,.5);ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][idx]
        masks={'all':np.ones(len(labels),bool)}
        if split=='primary':masks.update(first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10))
        else:masks.update(saturday=np.repeat(ts<1737849600,10),sunday=np.repeat(ts>=1737849600,10))
        metrics={name:dict(baseline_auc=float(roc_auc_score(labels[mask],base[mask])),candidate_auc=float(roc_auc_score(labels[mask],candidate[mask])),gain=float(roc_auc_score(labels[mask],candidate[mask])-roc_auc_score(labels[mask],base[mask]))) for name,mask in masks.items()};ll=float(log_loss(labels,candidate)-log_loss(labels,base));positive=all(v['gain']>0 for v in metrics.values());results[split]=dict(metrics=metrics,logloss_change=ll,all_slices_positive=positive,nonworse_logloss=ll<=0);np.savez_compressed(OUT/f'{split}_evidence.npz',indices=idx,labels=labels,users=e['users'],period=e['period'],baseline=base,candidate=candidate,delta=delta);print(split,results[split],flush=True)
    results['date_confirmation_passed']=all(r['all_slices_positive'] and r['nonworse_logloss'] for r in results.values());results['scope']='Reused V136 proxies, not exact V136 heldout predictions';m.write(OUT/'confirmation.json',results)
if __name__=='__main__':main()
