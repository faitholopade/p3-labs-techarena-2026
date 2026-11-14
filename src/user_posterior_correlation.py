"""Approximate cross-factor user correlations from pre-cutoff likelihood curvature.

This is a Fisher-based approximation, not an exact posterior. Its correlation
matrix can be combined with the existing fitted marginal standard deviations.
"""
import argparse,json,math,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from common import load_data,split_data,subset
from cluster_unit import warmup
from negative import pooled,sample
from uncertainty import torch_direction


@torch.no_grad()
def build(state,fit,pool,seed=20261027):
    rng=np.random.default_rng(seed);torch.set_num_threads(2)
    im=state['item.weight'];il=state['item_logstd.weight'].clamp(-3,1).exp()
    iv=torch_direction(im,il)
    um=state['user.weight'];un=um.norm(dim=1).clamp_min(1e-8);uv=um/un[:,None]
    q=state['exposure_items'];em=state['exposure_user.weight']
    beta=state['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=state['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()
    latent_size=8+q.shape[1]
    h=torch.diag(torch.tensor([.25]*8+[.5]*q.shape[1])).repeat(len(um),1,1)
    wu,wi,_=warmup(fit)
    for phase in ['real','warm']:
        count=len(fit['user']) if phase=='real' else len(wu)
        for st in range(0,count,2048):
            sl=slice(st,st+2048)
            if phase=='real':c=torch.tensor(fit['items'][sl]);u=torch.tensor(fit['user'][sl])
            else:c=torch.tensor(np.c_[wi[sl],sample(pool,wu[sl],rng)]);u=torch.tensor(wu[sl])
            dot=(iv[c]*uv[u,None]).sum(-1)
            z=beta*dot+eb*(q[c]*em[u,None]/math.sqrt(8)).sum(-1)+state['item_bias.weight'][c,0]+state['outside.weight'][u]
            p=torch.softmax(torch.cat([z,torch.zeros(len(u),1)],1),1)[:,:10] if phase=='real' else torch.softmax(z,1)
            pref=beta*(iv[c]-dot[:,:,None]*uv[u,None])/un[u,None,None]
            exposure=eb*q[c]/math.sqrt(8);grad=torch.cat([pref,exposure],2)
            mean=torch.einsum('nk,nkd->nd',p,grad)
            fisher=torch.einsum('nk,nkd,nke->nde',p,grad,grad)-mean[:,:,None]*mean[:,None,:]
            h.index_add_(0,u,fisher)
        print({'fisher_phase':phase,'events':count},flush=True)
    result=torch.empty_like(h)
    for st in range(0,len(h),1024):
        covariance=torch.linalg.inv(h[st:st+1024]);sd=covariance.diagonal(dim1=1,dim2=2).clamp_min(1e-8).sqrt()
        corr=covariance/(sd[:,:,None]*sd[:,None,:]);corr=(corr+corr.transpose(1,2))/2
        # Retain a small independent component for stable Cholesky sampling.
        result[st:st+1024]=.999*corr+.001*torch.eye(latent_size)[None]
    assert torch.isfinite(result).all()
    torch.linalg.cholesky(result)
    return result.numpy()


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--factor',required=True);p.add_argument('--cut',required=True,type=int);p.add_argument('--exposure',required=True);p.add_argument('--pool-cache',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    d,_,audit=load_data(Path(a.root)/'data');fi,_=split_data(d,a.cut);fit=subset(d,fi);del d
    state=torch.load(a.factor,map_location='cpu',weights_only=False)['state']
    pool=pooled(fit,audit['n_items'],audit['n_users'],a.cut,basis=a.exposure,cache=a.pool_cache)
    start=time.time();corr=build(state,fit,pool);out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.save(out,corr)
    report={'args':vars(a),'shape':list(corr.shape),'seconds':time.time()-start,'method':'Linearized real-slate and sampled warm-up Fisher curvature, fixed prior precision 0.25/0.5, normalized to a correlation matrix. Marginal variances remain those of the fitted model.'}
    out.with_suffix('.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':main()
