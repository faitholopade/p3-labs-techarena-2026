"""Regularized probabilistic topic model of observed click sequences."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from common import *

def counts(d,ni,nu):
    mask=d['hist']>0;u=np.broadcast_to(d['user'][:,None],mask.shape)[mask];i=d['hist'][mask];t=d['hist_ts'][mask];hit=d['target']<10
    u=np.r_[u,d['user'][hit]];i=np.r_[i,d['items'][hit,d['target'][hit]]];t=np.r_[t,d['timestamp'][hit]]
    t0=t.min();span=t.max()-t0+1;keys=(u.astype(np.int64)*(ni+1)+i)*span+(t-t0);keys=np.unique(keys);pairs=keys//span;rr=pairs//(ni+1);cc=pairs%(ni+1)
    return sp.coo_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(nu+1,ni+1)).tocsr()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--k',type=int,default=16);ap.add_argument('--iterations',type=int,default=60);ap.add_argument('--out',required=True);ap.add_argument('--alpha',type=float,default=.5);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cache='v2/P3_Labs_model_and_code/cache';d,e,audit=load_data(cache);fi,vi=split_data(d,1737849600);fit=subset(d,fi);val=subset(d,vi);ni=audit['n_items'];nu=audit['n_users'];s=stats(fit,ni,nu);m=counts(fit,ni,nu);co=m.tocoo();rr=co.row;cc=co.col;weight=co.data;pop=np.asarray(m.sum(0)).ravel();up=np.asarray(m.sum(1)).ravel()
    labels=np.load(cache+'/features2_1737849600/clusters.npz')[f'labels_{a.k}'];phi=np.full((ni+1,a.k),.01,dtype=np.float32);phi[np.arange(1,ni+1),labels[1:]]+=1;phi*=np.maximum(pop[:,None],.1);phi/=phi.sum(0,keepdims=True)
    theta=np.asarray(m@(phi/np.maximum(phi.sum(1,keepdims=True),1e-9)))+a.alpha;theta/=theta.sum(1,keepdims=True)
    for epoch in range(a.iterations):
        st=time.time();r=theta[rr]*phi[cc];r*=weight[:,None]/np.maximum(r.sum(1,keepdims=True),1e-20)
        nt=np.empty_like(theta);np_=np.empty_like(phi)
        for k in range(a.k):nt[:,k]=np.bincount(rr,weights=r[:,k],minlength=nu+1)+a.alpha;np_[:,k]=np.bincount(cc,weights=r[:,k],minlength=ni+1)+50*pop/max(pop.sum(),1)+1e-8
        theta=nt/nt.sum(1,keepdims=True);phi=np_/np_.sum(0,keepdims=True)
        if (epoch+1)%10==0:print('EM',epoch+1,'seconds',time.time()-st,flush=True)
    np.savez_compressed(out/'model.npz',theta=theta,phi=phi,pop=pop)
    pc=np.load('v2/experiments/n32/valid_pred.npy').sum(1,keepdims=True);y=binary_labels(val);ph=phi[val['hist']];mask=val['hist']>0;th=np.full((len(y),a.k),1/a.k,np.float32)
    for _ in range(12):
        r=ph*th[:,None,:];r/=np.maximum(r.sum(2,keepdims=True),1e-20);th=(r*mask[:,:,None]).sum(1)+a.alpha;th/=th.sum(1,keepdims=True)
    best=(0,None)
    for mix in [0,.5,1]:
        tt=mix*th+(1-mix)*theta[val['user']];p0=np.einsum('nkd,nd->nk',phi[val['items']],tt)
        for correct in [0,.5,1]:
            z=np.log(np.maximum(p0,1e-20))-correct*np.log(np.maximum(pop[val['items']],1))
            for temp in [.5,1,2]:
                pred=softmax(z*temp,axis=1)*pc;auc=roc_auc_score(y.ravel(),pred.ravel())
                if auc>best[0]:best=(auc,(mix,correct,temp));np.save(out/'valid_pred.npy',pred)
    print('BEST',best,flush=True);(out/'metrics.json').write_text(json.dumps({'auc':best[0],'configuration':best[1],'args':vars(a)},indent=2))

if __name__=='__main__':main()
