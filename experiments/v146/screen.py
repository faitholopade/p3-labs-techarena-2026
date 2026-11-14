"""Evaluate fixed blends of complete causal LightGBM predictors."""
import json,time
from pathlib import Path
import numpy as np,lightgbm as lgb
from scipy.special import expit,logit,softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v146'
def blend(base,alternative,mode,weight):
    if mode=='logit':return expit((1-weight)*logit(base)+weight*logit(alternative.clip(1e-7,1-1e-7)))
    b=base.reshape(-1,10);a=alternative.reshape(-1,10);return (softmax((1-weight)*np.log(b)+weight*np.log(a.clip(1e-10)),axis=1)*b.sum(1,keepdims=True)).ravel()
def main():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');f=np.load(ROOT/'runtime/v145/forecast.npz');x=f['X'];width=int(f['control_width']);assert np.array_equal(e['indices'],f['indices']);base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={};models={};standalone={}
    for name in ['control7','latent7','control15','latent15']:
        model=lgb.Booster(model_file=str(ROOT/f'runtime/v145/cohort0/{name}.txt'));a=expit(f['init']+model.predict(x[:,:width] if name.startswith('control') else x,raw_score=True,num_threads=3));models[name]=a;standalone[name]={k:float(roc_auc_score(y[z],a[z])) for k,z in masks.items()}
        for mode,weights in [('logit',[.1,.25,.5]),('choice',[.25,.5])]:
            for weight in weights:
                p=blend(base,a,mode,weight);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(model=name,mode=mode,weight=weight,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{name}_{mode}_{weight}']=p;print(row,flush=True)
    (OUT/'primary_screen.json').write_text(json.dumps(dict(results=rows,baseline_auc=ba,standalone_auc=standalone),indent=2)+'\n');np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred);np.savez_compressed(OUT/'primary_model_predictions.npz',indices=e['indices'],**models)
if __name__=='__main__':main()
