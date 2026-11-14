"""Paired user-Poisson bootstrap of AUC change, preserving score ties."""
import argparse,json,time
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

def prepare(y,p,user):
    order=np.argsort(p,kind='stable');s=p[order];starts=np.r_[0,np.flatnonzero(s[1:]!=s[:-1])+1]
    return y[order].astype(np.float64),user[order],starts

def auc(prepared,weight):
    y,u,starts=prepared;w=weight[u];positive=np.add.reduceat(w*y,starts);negative=np.add.reduceat(w*(1-y),starts)
    return float(np.dot(positive,np.cumsum(negative)-.5*negative)/(positive.sum()*negative.sum()))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--evidence',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);a=parser.parse_args()
    d=np.load(a.evidence);y=d['labels'];u=d['users'];_,codes=np.unique(u,return_inverse=True);n=codes.max()+1;b=prepare(y,d['baseline'],codes);c=prepare(y,d['candidate'],codes)
    one=np.ones(n);bp=roc_auc_score(y,d['baseline']);cp=roc_auc_score(y,d['candidate']);assert abs(auc(b,one)-bp)<1e-12 and abs(auc(c,one)-cp)<1e-12
    rng=np.random.default_rng(20261101);began=time.time();g=[]
    for i in range(400):
        w=rng.poisson(1,n);g.append(auc(c,w)-auc(b,w))
    report={'replicates':400,'seed':20261101,'unit':'user','method':'paired Poisson weights with exact tie-aware weighted AUC','users':int(n),'baseline_auc':bp,'candidate_auc':cp,'point_gain':cp-bp,'interval_95':np.quantile(g,[.025,.975]).tolist(),'positive_replicates':int((np.array(g)>0).sum()),'lower_bound_positive':bool(np.quantile(g,.025)>0),'seconds':time.time()-began,'development_reuse_limit':'No multiple-search correction; not an untouched holdout.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');np.save(a.out.with_suffix('.npy'),g);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
