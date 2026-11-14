"""Refine the joint factors with an additional global candidate-ranking objective.

Fits only pre-cutoff evidence, at a fixed schedule with no checkpoint selection.
"""
from common import *
from pathlib import Path
from residual_model import load
from cluster_unit import warmup
from negative import pooled,sample as sample_pool
import torch,time,math,argparse
from torch import nn

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cut',type=int,required=True)
    ap.add_argument('--root',default='.')
    ap.add_argument('--exposure',required=True)
    ap.add_argument('--pool-cache',default='scratch_pools')
    ap.add_argument('--init',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=8819)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--threads',type=int,default=2)
    ap.add_argument('--preference-kl',type=float,default=.25)
    ap.add_argument('--exposure-kl',type=float,default=.5)
    ap.add_argument('--rank-weight',type=float,default=.5)
    a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed)
    d,_,audit=load_data(Path(a.root)/'data');fi,_=split_data(d,a.cut);fit=subset(d,fi)
    net=load(a.init)
    for p in net.parameters():p.requires_grad_(True)
    opt=torch.optim.Adam(net.parameters(),lr=a.lr)
    wu,wi,_=warmup(fit);N=len(fi)+len(wu)
    pool=pooled(fit,audit['n_items'],audit['n_users'],a.cut,
        basis=a.exposure,cache=a.pool_cache)
    history=[]
    for epoch in range(1,a.epochs+1):
        start=time.time();net.train()
        opt.param_groups[0]['lr']=a.lr*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        order=rng.permutation(N);losses=[]
        for st in range(0,N,2048):
            ix=order[st:st+2048];tr=ix[ix<len(fi)];wa=ix[ix>=len(fi)]-len(fi)
            nc=sample_pool(pool,wu[wa],rng)
            c=torch.tensor(np.concatenate([fit['items'][tr],np.c_[wi[wa],nc]]))
            u=torch.tensor(np.r_[fit['user'][tr],wu[wa]])
            z=net(c,u,True);nt=len(tr)
            real=nn.functional.cross_entropy(torch.cat([z[:nt],torch.zeros(nt,1)],1),torch.tensor(fit['target'][tr]),reduction='sum')
            warm=nn.functional.cross_entropy(z[nt:],torch.zeros(len(wa),dtype=torch.long),reduction='sum')
            mu=net.exposure_user.weight;ls=net.exposure_logstd.weight.clamp(-4,1)
            extra=.5*(mu.square().sum()+8*(ls.mul(2).exp()-1-2*ls).sum())+10*net.item_bias.weight.square().sum()
            loss=(real+warm)/len(ix)+(a.preference_kl*net.kl()+a.exposure_kl*extra)/N
            prob=torch.softmax(torch.cat([z[:nt],torch.zeros(nt,1)],1),1)[:,:10].ravel().clamp(1e-6,1-1e-6)
            logits=torch.logit(prob);target=torch.tensor(fit['target'][tr])
            label=(torch.arange(10)[None,:]==target[:,None]).ravel()
            positive=logits[label];negative=logits[~label]
            if len(positive) and len(negative):
                sampled=negative[torch.randint(len(negative),(len(positive),32))]
                loss=loss+a.rank_weight*(nt/len(ix))*nn.functional.softplus(sampled-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
        rec={'epoch':epoch,'seconds':time.time()-start,'loss':float(np.mean(losses))}
        history.append(rec);print(rec,flush=True)
    torch.save({'state':net.state_dict(),'cut':a.cut,'epoch':a.epochs},out/'residual.pt')
    (out/'training.json').write_text(json.dumps({'args':vars(a),'training_events':len(fi),'warmup_clicks':len(wu),'records':history},indent=2))

if __name__=='__main__':main()
