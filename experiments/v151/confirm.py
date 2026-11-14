"""Score one frozen complementary correction, without weight tuning."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v151'
def delta(correlation,context):
    d=np.log(correlation['1.0'][:,:10]/correlation['0.0'][:,:10]);d-=d.mean(1,keepdims=True);return .3*np.tanh(d/.3)+.25*.3*np.tanh(context['k4_p32']/.3)
def score(split):
    e=np.load(ROOT/f'runtime/v142/{split}_evidence.npz');idx=e['indices'];where={int(x):j for j,x in enumerate(idx)};change=np.full((len(idx),10),np.nan)
    for cohort in ([0] if split=='primary' else [1,2]):
        c=np.load(ROOT/f'runtime/v147/cohort{cohort}/predictions.npz');t=np.load(ROOT/f'runtime/v149/cohort{cohort}/predictions.npz');assert np.array_equal(c['indices'],t['indices']);j=np.array([where[int(x)] for x in c['indices']]);change[j]=delta(c,t)
    assert np.isfinite(change).all();base=e['candidate'];b=base.reshape(-1,10);candidate=(softmax(np.log(b)+change,axis=1)*b.sum(1,keepdims=True)).ravel();y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][idx];masks={'all':np.ones(len(y),bool)}
    if split=='primary':masks.update(first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10))
    else:masks.update(saturday=np.repeat(ts<1737849600,10),sunday=np.repeat(ts>=1737849600,10))
    metrics={k:{'baseline_auc':float(roc_auc_score(y[m],base[m])),'candidate_auc':float(roc_auc_score(y[m],candidate[m])),'gain':float(roc_auc_score(y[m],candidate[m])-roc_auc_score(y[m],base[m]))} for k,m in masks.items()};ll=float(log_loss(y,candidate)-log_loss(y,base));report=dict(metrics=metrics,logloss_change=ll,date_checks_passed=all(v['gain']>0 for v in metrics.values()) and ll<=0,normal_release_minimum_passed=False,scope='Reused V142 proxies, adaptive fixed combination');np.savez_compressed(OUT/f'{split}_evidence.npz',indices=idx,labels=y,users=e['users'],period=e['period'],baseline=base,candidate=candidate,delta=change);(OUT/f'{split}_confirmation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--split',choices=['primary','secondary'],required=True);a=p.parse_args();score(a.split)
