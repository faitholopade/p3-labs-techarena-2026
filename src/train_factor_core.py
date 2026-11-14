"""Fixed-schedule chronological preference and residual fits, without validation selection."""
from common import *
from variational import Variational
from residual_model import Residual
from cluster_unit import warmup
from negative import pooled,sample as sample_pool
import argparse,torch,time,math
from torch import nn

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--init',required=True);ap.add_argument('--exposure',required=True);ap.add_argument('--pool-cache',default='scratch_pools');ap.add_argument('--cut',type=int,required=True);ap.add_argument('--out',required=True);ap.add_argument('--seed',type=int,default=3951);ap.add_argument('--kl',type=float,default=.25);ap.add_argument('--warm-weight',type=float,default=1.);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3);d,_,audit=load_data(Path(a.root)/'data');fi,_=split_data(d,a.cut);fit=subset(d,fi);ni=audit['n_items'];nu=audit['n_users'];m=interaction_matrix(fit,ni,nu);ic=np.asarray(m.sum(0)).ravel();uc=np.asarray(m.sum(1)).ravel();s=stats(fit,ni,nu);outside=(logit(np.clip((s['uh']+6.3)/(s['ui']+10),1e-4,1-1e-4))-np.log(10)-.6).astype(np.float32);v=np.load(a.init)[:,:8].copy();v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);u=m@v;u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8);wu,wi,_=warmup(fit);N=len(fi)+len(wu);effective_n=len(fi)+a.warm_weight*len(wu);pool=pooled(fit,ni,nu,a.cut,basis=a.exposure,cache=a.pool_cache);records=[]
    torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);net=Variational(v,u,ic,uc,outside);opt=torch.optim.Adam(net.parameters(),lr=.006)
    for epoch in range(1,17):
        start=time.time();lr=.0006+.0054*.5*(1+math.cos(math.pi*(epoch-1)/23));opt.param_groups[0]['lr']=lr;order=rng.permutation(N);losses=[]
        for st in range(0,N,2048):
            ix=order[st:st+2048];tr=ix[ix<len(fi)];wa=ix[ix>=len(fi)]-len(fi);c=torch.tensor(fit['items'][tr]);u0=torch.tensor(fit['user'][tr]);wu0=torch.tensor(wu[wa]);wi0=torch.tensor(wi[wa]);nc=torch.tensor(sample_pool(pool,wu[wa],rng));z=net(c,u0,True);rl=nn.functional.cross_entropy(torch.cat([z,torch.zeros(len(tr),1)],1),torch.tensor(fit['target'][tr]),reduction='sum');uv=net.vectors('user',wu0,True);pos=(net.vectors('item',wi0,True)*uv).sum(-1);neg=torch.einsum('nkd,nd->nk',net.vectors('item',nc,True),uv);beta=net.log_beta.clamp(math.log(2),math.log(8)).exp();zw=torch.cat([pos[:,None],neg],1)*beta;wl=nn.functional.cross_entropy(zw,torch.zeros(len(wa),dtype=torch.long),reduction='sum');loss=(rl+a.warm_weight*wl)/len(ix)*N/effective_n+a.kl*net.kl()/effective_n;opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
        rec={'phase':'preference','epoch':epoch,'seconds':time.time()-start,'loss':float(np.mean(losses))};records.append(rec);print(rec,flush=True)
    torch.save({'state':net.state_dict(),'cut':a.cut,'epoch':16},out/'preference.pt')
    torch.manual_seed(a.seed+529);rng=np.random.default_rng(a.seed+529);rn=Residual(v,u,ic,uc,outside);rn.load_state_dict(net.state_dict());q=np.load(a.exposure)['v'][:,1:9].copy();q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);q[0]=0;ex=sp.coo_matrix((np.ones(len(fi)*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=m.shape).tocsr();ne=np.asarray(ex.sum(1)).ravel();mu=(m@q/np.maximum(uc[:,None],1)-ex@q/np.maximum(ne[:,None],1))*2;rn.attach(q,mu.astype(np.float32));opt=torch.optim.Adam([p for p in rn.parameters() if p.requires_grad],lr=.01)
    for epoch in range(1,18):
        start=time.time();order=rng.permutation(N);losses=[]
        for st in range(0,N,4096):
            ix=order[st:st+4096];tr=ix[ix<len(fi)];wa=ix[ix>=len(fi)]-len(fi);nc=sample_pool(pool,wu[wa],rng);c=torch.tensor(np.concatenate([fit['items'][tr],np.c_[wi[wa],nc]]));u0=torch.tensor(np.r_[fit['user'][tr],wu[wa]]);z=rn(c,u0,True);nt=len(tr);rl=nn.functional.cross_entropy(torch.cat([z[:nt],torch.zeros(nt,1)],1),torch.tensor(fit['target'][tr]),reduction='sum');wl=nn.functional.cross_entropy(z[nt:],torch.zeros(len(wa),dtype=torch.long),reduction='sum');mu=rn.exposure_user.weight;ls=rn.exposure_logstd.weight.clamp(-4,1);kl=.5*(mu.square().sum()+8*(ls.mul(2).exp()-1-2*ls).sum())+10*rn.item_bias.weight.square().sum();loss=(rl+a.warm_weight*wl)/len(ix)*N/effective_n+.5*kl/effective_n;opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
        rec={'phase':'residual','epoch':epoch,'seconds':time.time()-start,'loss':float(np.mean(losses))};records.append(rec);print(rec,flush=True)
    torch.save({'state':rn.state_dict(),'cut':a.cut,'epoch':17},out/'residual.pt');(out/'training.json').write_text(json.dumps({'args':vars(a),'training_events':len(fi),'warmup_clicks':len(wu),'records':records},indent=2))
if __name__=='__main__':main()
