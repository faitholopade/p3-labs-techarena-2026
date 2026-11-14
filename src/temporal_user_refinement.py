"""Refine user factors with fixed catalog factors and recency-weighted real slates."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np,scipy.sparse as sp,torch
from torch import nn
from sklearn.linear_model import Ridge
from common import load_data,subset
from residual_model import load
from cluster_unit import warmup
from negative import pooled,sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','init','exposure','pool-cache','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--half-life-days',type=float,default=0)
    p.add_argument('--epochs',type=int,default=12);p.add_argument('--lr',type=float,default=.0015)
    p.add_argument('--seed',type=int,default=20261110);p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    if a.half_life_days<0:raise ValueError('Use zero for the no-decay control.')
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    data,_,audit=load_data(a.root/'data');fit=subset(data,np.flatnonzero(data['timestamp']<a.cut));del data
    assert fit['timestamp'].max()<a.cut
    net=load(a.init);allowed={'user','user_logstd','exposure_user','exposure_logstd','outside'}
    frozen={name:value.detach().clone() for name,value in net.state_dict().items() if name.split('.')[0] not in allowed}
    for name,param in net.named_parameters():param.requires_grad_(name.split('.')[0] in allowed)
    nu,ni=audit['n_users']+1,audit['n_items']+1;count=np.bincount(fit['user'],minlength=nu)
    imp=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=(nu,ni)).tocsr()
    profile=np.asarray(imp@net.exposure_items.numpy())/np.maximum(10*count[:,None],1)
    dense=np.flatnonzero(count>=10);ridge=Ridge(alpha=10).fit(profile[dense],net.exposure_user.weight.detach().numpy()[dense])
    prior=torch.tensor(ridge.predict(profile),dtype=torch.float32);del imp,profile
    weights=np.ones(len(fit['user']),np.float32)
    if a.half_life_days:
        weights=np.exp2(-(a.cut-fit['timestamp'])/(86400*a.half_life_days)).astype(np.float32);weights/=weights.mean()
    weights=torch.tensor(weights);wu,wi,wx=warmup(fit);total=len(fit['user'])+len(wu)
    pool=pooled(fit,ni-1,nu-1,a.cut,basis=a.exposure,cache=a.pool_cache)
    opt=torch.optim.Adam([p for p in net.parameters() if p.requires_grad],lr=a.lr);records=[]
    print({'real_events':len(fit['user']),'warm_events':len(wu),'half_life_days':a.half_life_days,'real_weight_range':[float(weights.min()),float(weights.max())]},flush=True)
    for epoch in range(1,a.epochs+1):
        started=time.time();net.train();opt.param_groups[0]['lr']=a.lr*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        order=rng.permutation(total);losses=[]
        for start in range(0,total,2048):
            ix=order[start:start+2048];tr=ix[ix<len(fit['user'])];wa=ix[ix>=len(fit['user'])]-len(fit['user']);nt=len(tr)
            c=torch.tensor(np.concatenate([fit['items'][tr],np.c_[wi[wa],sample(pool,wu[wa],rng)]]))
            users=torch.tensor(np.r_[fit['user'][tr],wu[wa]]);context=torch.tensor(np.r_[fit['context'][tr],wx[wa]])
            state=torch.get_rng_state();z1=net(c,users,True,context);torch.set_rng_state(state);z2=net(c,users,-1,context)
            draws=[z1,z2];target=torch.tensor(fit['target'][tr]);warm_target=torch.zeros(len(wa),dtype=torch.long)
            real=sum((nn.functional.cross_entropy(torch.cat([z[:nt],torch.zeros(nt,1)],1),target,reduction='none')*weights[tr]).sum() for z in draws)/2
            warm=sum(nn.functional.cross_entropy(z[nt:],warm_target,reduction='sum') for z in draws)/2
            mu=net.exposure_user.weight;ls=net.exposure_logstd.weight.clamp(-4,1)
            extra=.5*((mu-prior).square().sum()/.5+(mu.shape[-1]//ls.shape[-1])*((2*ls).exp()/.5-1+math.log(.5)-2*ls).sum())
            extra=extra+10*net.item_bias.weight.square().sum()
            loss=(real+2*warm)/len(ix)+(.25*net.kl()+.5*extra)/total
            # Keep the historical ranking term for a controlled recency comparison.
            prob=torch.stack([torch.softmax(torch.cat([z[:nt],torch.zeros(nt,1)],1),1)[:,:10] for z in draws]).mean(0).ravel().clamp(1e-6,1-1e-6)
            labels=(torch.arange(10)[None,:]==target[:,None]).ravel();logits=torch.logit(prob);positive=logits[labels];negative=logits[~labels]
            if len(positive) and len(negative):
                sampled=negative[torch.randint(len(negative),(len(positive),32))]
                positive_weight=weights[tr].repeat_interleave(10)[labels]
                loss+=.5*(nt/len(ix))*(nn.functional.softplus(sampled-positive[:,None]).mean(1)*positive_weight).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(float(loss.detach()))
        rec={'epoch':epoch,'seconds':time.time()-started,'loss':float(np.mean(losses))};records.append(rec);print(rec,flush=True)
    for name,value in frozen.items():
        if not torch.equal(net.state_dict()[name],value):raise ValueError('Frozen catalog/global parameter changed: '+name)
    with a.init.open('rb') as f:source_sha=hashlib.file_digest(f,'sha256').hexdigest()
    config={k:(str(v) if isinstance(v,Path) else v) for k,v in vars(a).items()}
    report={'config':config,'initial_sha256':source_sha,'records':records,'catalog_and_global_parameters_unchanged':True,
            'max_training_timestamp':int(fit['timestamp'].max()),'warm_weight':2.,'warm_history_recency_weighted':False,
            'real_weights_normalized_to_mean_one':True,'query_outcomes_used':False,
            'description':'Recency-weighted likelihood refinement of user preference, exposure, uncertainty and event intercept; fixed catalog and global coefficients. Zero half-life is a continued-fitting control.'}
    torch.save({'state':net.state_dict(),'cut':a.cut,'epoch':a.epochs,'config':config},a.out/'model.pt')
    (a.out/'training.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
