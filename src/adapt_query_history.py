"""Infer previously unseen users' factors from supplied historical clicks only."""
import argparse,json,math
from pathlib import Path
import numpy as np,torch
from torch import nn
from common import load_data,subset
from residual_model import load
from negative import pooled,sample as sample_pool


def adapt(net,fit,query,pool,cut,mode='cluster',steps=120,seed=20261031,kl_weight=.25):
    torch.set_num_threads(2);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    user=np.asarray(query['user']);hist=np.asarray(query['hist']);hist_ts=np.asarray(query['hist_ts'])
    if len(np.unique(user))!=len(user):raise ValueError('Expected one supplied query history per user.')
    count=np.bincount(fit['user'],minlength=len(net.user.weight))
    mask=(count[user,None]==0)&(hist>0)&(hist_ts<cut)
    event,position=np.nonzero(mask);u=user[event];positive=hist[event,position];t=hist_ts[event,position]
    # A history timestamp is used once even if a malformed input repeats it.
    records=np.rec.fromarrays([u,positive,t],names='u,i,t');_,keep=np.unique(records,return_index=True)
    u=u[keep];positive=positive[keep];t=t[keep];event=event[keep]
    unique=np.unique(u);original_user=net.user.weight.detach().clone();original_std=net.user_logstd.weight.detach().clone()
    if not len(u):return net,{'users_adapted':0,'new_history_clicks':0}
    pool={k:v.copy() for k,v in pool.items()}
    if mode=='cluster':
        q=net.exposure_items.numpy();centers=[]
        for g in range(len(pool['ptr'])-1):centers.append(q[pool['pool'][pool['ptr'][g]:pool['ptr'][g+1]]].mean(0))
        centers=np.asarray(centers);centers/=np.maximum(np.linalg.norm(centers,axis=1,keepdims=True),1e-8)
        profile=q[query['items']].mean(1);profile/=np.maximum(np.linalg.norm(profile,axis=1,keepdims=True),1e-8)
        pool['labels'][user]=np.argmax(profile@centers.T,1)
    for p in net.parameters():p.requires_grad_(False)
    net.user.weight.requires_grad_(True);net.user_logstd.weight.requires_grad_(True)
    optimizer=torch.optim.Adam([net.user.weight,net.user_logstd.weight],lr=.03)
    context=torch.tensor(t%86400//21600+4*((t//86400+3)%7>=5));users=torch.tensor(u)
    losses=[]
    for step in range(steps):
        optimizer.param_groups[0]['lr']=.03*(.1+.9*.5*(1+math.cos(math.pi*step/max(1,steps-1))))
        if mode=='slate':negative=np.take_along_axis(query['items'][event],rng.integers(0,10,(len(u),9)),axis=1)
        else:negative=sample_pool(pool,u,rng)
        candidates=torch.tensor(np.c_[positive,negative]);state=torch.get_rng_state();z1=net(candidates,users,True,context);torch.set_rng_state(state);z2=net(candidates,users,-1,context)
        target=torch.zeros(len(u),dtype=torch.long);likelihood=(nn.functional.cross_entropy(z1,target)+nn.functional.cross_entropy(z2,target))/2
        mu=net.user.weight[unique];ls=net.user_logstd.weight[unique].clamp(-3,1)
        kl=.5*(mu.square().sum()+8*((2*ls).exp()-1-2*ls).sum())
        loss=2*likelihood+kl_weight*kl/len(u);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step();losses.append(float(loss.detach()))
    unaffected=np.ones(len(net.user.weight),bool);unaffected[unique]=False
    assert torch.equal(net.user.weight.detach()[unaffected],original_user[unaffected])
    assert torch.equal(net.user_logstd.weight.detach()[unaffected],original_std[unaffected])
    return net,{'users_adapted':len(unique),'new_history_clicks':len(u),'steps':steps,'seed':seed,'kl_weight':kl_weight,'negative_mode':mode,'first_loss':losses[0],'last_loss':losses[-1],'unaffected_user_parameters_identical':True,'query_outcomes_used':False,'history_before_training_cutoff':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','factor','query','exposure','pool-cache','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--mode',choices=['cluster','slate'],default='cluster')
    p.add_argument('--steps',type=int,default=120);p.add_argument('--seed',type=int,default=20261031)
    p.add_argument('--kl-weight',type=float,default=.25)
    a=p.parse_args();d,_,audit=load_data(a.root/'data');fit=subset(d,np.flatnonzero(d['timestamp']<a.cut));query=dict(np.load(a.query))
    net=load(a.factor);pool=pooled(fit,audit['n_items'],audit['n_users'],a.cut,basis=a.exposure,cache=a.pool_cache)
    net,report=adapt(net,fit,query,pool,a.cut,a.mode,a.steps,a.seed,a.kl_weight);a.out.mkdir(parents=True,exist_ok=True)
    ck=torch.load(a.factor,map_location='cpu',weights_only=False);ck['state']=net.state_dict();torch.save(ck,a.out/'residual.pt')
    (a.out/'adaptation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
