"""Learn user-posterior correlations while preserving all fitted means and variances."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,torch
from torch import nn
from common import load_data,subset
from residual_model import load
from cluster_unit import warmup
from negative import pooled,sample


def lower_factor(raw,dimension=16):
    rows,cols=torch.tril_indices(dimension,dimension,offset=-1,device=raw.device)
    result=torch.eye(dimension,device=raw.device,dtype=raw.dtype).expand(len(raw),-1,-1).clone();result[:,rows,cols]=raw
    norm=result.square().sum(2).sqrt();return result/norm[:,:,None],torch.log(norm).sum(1)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','init','exposure','pool-cache','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--penalty',type=float,default=.5)
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--threads',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(20261113);rng=np.random.default_rng(20261113)
    data,_,audit=load_data(a.root/'data');fit=subset(data,np.flatnonzero(data['timestamp']<a.cut));del data
    original=torch.load(a.init,map_location='cpu',weights_only=False);assert int(original['cut'])==a.cut;net=load(a.init)
    for param in net.parameters():param.requires_grad_(False)
    dimension=net.dim+net.exposure_user.weight.shape[1]
    if dimension!=16:raise ValueError('Expected eight preference and eight exposure coordinates.')
    wu,wi,wx=warmup(fit);total=len(fit['user'])+len(wu);nu=len(net.user.weight)
    count=torch.from_numpy(np.bincount(np.r_[fit['user'],wu],minlength=nu).clip(1).astype(np.float32))
    pool=pooled(fit,audit['n_items'],audit['n_users'],a.cut,basis=a.exposure,cache=a.pool_cache)
    correlation=nn.Embedding(nu,dimension*(dimension-1)//2,sparse=True);nn.init.zeros_(correlation.weight)
    optimizer=torch.optim.SparseAdam(correlation.parameters(),lr=.008);average=torch.zeros_like(correlation.weight);navg=0;records=[]
    beta=net.log_beta.clamp(math.log(2),math.log(8)).exp();eb=net.log_exposure_beta.clamp(math.log(.5),math.log(8)).exp()
    print({'events':len(fit['user']),'warm_events':len(wu),'dimension':dimension,'penalty':a.penalty},flush=True)
    for epoch in range(1,a.epochs+1):
        started=time.time();order=rng.permutation(total);losses=[]
        optimizer.param_groups[0]['lr']=.008*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        for st in range(0,total,2048):
            ix=order[st:st+2048];tr=ix[ix<len(fit['user'])];wa=ix[ix>=len(fit['user'])]-len(fit['user']);nt=len(tr)
            c=torch.tensor(np.concatenate([fit['items'][tr],np.c_[wi[wa],sample(pool,wu[wa],rng)]]));u=torch.tensor(np.r_[fit['user'][tr],wu[wa]]);context=torch.tensor(np.r_[fit['context'][tr],wx[wa]])
            chol,kl_corr=lower_factor(correlation(u),dimension);noise=torch.einsum('nij,nj->ni',chol,torch.randn(len(u),dimension));item_noise=torch.randn(len(u),10,net.dim);draws=[]
            for sign in [1.,-1.]:
                uv=nn.functional.normalize(net.user(u)+sign*net.user_logstd(u).clamp(-3,1).exp()*noise[:,:8],dim=-1)
                iv=nn.functional.normalize(net.item(c)+sign*net.item_logstd(c).clamp(-3,1).exp()*item_noise,dim=-1)
                eu=(net.exposure_user(u)+sign*net.exposure_logstd(u).clamp(-4,1).exp()*noise[:,8:])/math.sqrt(8)
                outside=net.outside(u)
                if hasattr(net,'context_matrix'):
                    uv=uv+torch.einsum('nd,ndk->nk',uv,net.context_matrix[context])+net.context_vector(context);outside=outside+net.context_outside(context)
                z=(iv*uv[:,None]).sum(-1)*beta+(net.exposure_items[c]*eu[:,None]).sum(-1)*eb+net.item_bias(c).squeeze(-1)+outside;draws.append(z)
            real=sum(nn.functional.cross_entropy(torch.cat([z[:nt],torch.zeros(nt,1)],1),torch.tensor(fit['target'][tr]),reduction='sum') for z in draws)/2
            warm=sum(nn.functional.cross_entropy(z[nt:],torch.zeros(len(wa),dtype=torch.long),reduction='sum') for z in draws)/2
            # Under event sampling, E[KL(user)/event_count(user)] = total KL / N.
            loss=(real+2*warm)/len(ix)+a.penalty*(kl_corr/count[u]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
        if epoch>=7:
            navg+=1;average.mul_((navg-1)/navg).add_(correlation.weight.detach(),alpha=1/navg)
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-started};records.append(rec);print(rec,flush=True)
    if not navg:average=correlation.weight.detach().clone()
    with torch.no_grad():
        chol,kl=lower_factor(average,dimension);cov=chol@chol.transpose(1,2)
        diag_error=float((cov.diagonal(dim1=1,dim2=2)-1).abs().max());assert diag_error<1e-5
        offdiag=cov-torch.eye(dimension)[None];mean_abs=float(offdiag.abs().sum()/(len(cov)*dimension*(dimension-1)))
    for key,value in net.state_dict().items():
        if not torch.equal(value,original['state'][key]):raise ValueError('Frozen factor changed.')
    with a.init.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    torch.save({'state':original['state'],'correlation_raw':average,'cut':a.cut,'penalty':a.penalty},a.out/'model.pt')
    report={'cut':a.cut,'initial_sha256':sha,'penalty':a.penalty,'epochs':a.epochs,'seed':20261113,'records':records,'marginal_variance_max_error':diag_error,'mean_absolute_offdiagonal_correlation':mean_abs,'means_and_marginal_variances_frozen':True,'max_training_timestamp':int(fit['timestamp'].max()),'query_outcomes_used':False,'limits':'Correlation-only variational likelihood refinement with a regularisation penalty; not an exact joint posterior. Ranking term omitted; this is a new inference approximation.'}
    (a.out/'training.json').write_text(json.dumps(report,indent=2)+'\n');print({k:report[k] for k in ['marginal_variance_max_error','mean_absolute_offdiagonal_correlation']},flush=True)


if __name__=='__main__':main()
