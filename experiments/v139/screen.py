"""Replay the rejected full-factor-continuation comparison."""
import importlib.util,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('v140_support',ROOT/'experiments/v140/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    e=np.load(ROOT/'runtime/v136/primary_evidence.npz');v=np.load(ROOT/'runtime/v138/cohort0/predictions.npz');old=np.load(ROOT/'runtime/v125/predictions.npz');assert np.array_equal(e['indices'],v['indices']) and np.array_equal(e['indices'],old['indices']);original=np.c_[old['original'],1-old['original'].sum(1)];base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[mask],base[mask]) for k,mask in masks.items()};rows=[];pred={}
    for arm in ['control','angular']:
        for mode in ['choice','joint']:
            for strength in [.25,.5,1.]:
                p=m.transfer(base,original,v[arm],mode,strength);g={k:float(roc_auc_score(y[mask],p[mask])-ba[k]) for k,mask in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));rows.append(dict(arm=arm,mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0));pred[f'{arm}_{mode}_{strength}']=p
    saved=np.load(ROOT/'runtime/v139/primary_predictions.npz');assert max(float(abs(saved[k]-v).max()) for k,v in pred.items())<1e-14;print(json.dumps(dict(recipes=len(rows),saved_prediction_replay=True)))
if __name__=='__main__':main()
