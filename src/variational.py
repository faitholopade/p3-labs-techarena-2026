"""Projected-normal variational preference factors with event choice likelihood.

Latent Gaussian vectors are normalized onto the unit sphere before scoring.
The loss uses reparameterized samples and an explicit full-dataset KL term.
"""
import sys,argparse,time,json,math
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(BASE/'v2/experiments'),str(BASE/'v3')]
from common import *
from latent import warmup
from negative import pooled,sample as sample_pool
import torch
from torch import nn
from scipy.special import softmax
from uncertainty import torch_direction

class Variational(nn.Module):
    def __init__(self,v,u,icount,ucount,outside,geometry='spherical',context_model=False):
        super().__init__();self.dim=v.shape[1];self.geometry=geometry
        self.item=nn.Embedding.from_pretrained(torch.tensor(v*2.5),freeze=False,padding_idx=0);self.user=nn.Embedding.from_pretrained(torch.tensor(u*2.5),freeze=False,padding_idx=0)
        ist=np.clip(2.5/np.sqrt(2*icount+1),.2,1).astype(np.float32);ust=np.clip(2.5/np.sqrt(2*ucount+1),.2,1).astype(np.float32)
        self.item_logstd=nn.Embedding.from_pretrained(torch.tensor(np.log(ist)[:,None]),freeze=False);self.user_logstd=nn.Embedding.from_pretrained(torch.tensor(np.log(ust)[:,None]),freeze=False);self.outside=nn.Embedding.from_pretrained(torch.tensor(outside[:,None]),freeze=False);self.log_beta=nn.Parameter(torch.tensor(math.log(4.)))
        if context_model:
            self.context_matrix=nn.Parameter(torch.zeros(8,self.dim,self.dim));self.context_vector=nn.Embedding(8,self.dim);nn.init.zeros_(self.context_vector.weight);self.context_outside=nn.Embedding(8,1);nn.init.zeros_(self.context_outside.weight)
        self.register_buffer('outside_prior',torch.full((len(u),),-2.4));self.register_buffer('initial',torch.tensor(v))
    def vectors(self,kind,ids,sample):
        mu=getattr(self,kind)(ids);std=getattr(self,kind+'_logstd')(ids).clamp(-3,1).exp()
        if self.geometry=='gaussian':return (mu+std*torch.randn_like(mu) if sample else mu)/math.sqrt(self.dim)
        if sample:
            noise=torch.randn_like(mu)
            if sample==-1:noise=-noise
            return nn.functional.normalize(mu+std*noise,dim=-1)
        return torch_direction(mu,std,exact=getattr(self,'exact_direction',False))
    def forward(self,c,u,sample=True,context=None,preference=None):
        v=self.vectors('item',c,sample);uv=self.vectors('user',u,sample) if preference is None else preference;offset=self.outside(u)
        if hasattr(self,'context_matrix') and context is not None:
            uv=uv+torch.einsum('nd,ndk->nk',uv,self.context_matrix[context])+self.context_vector(context);offset=offset+self.context_outside(context)
        z=(v*uv[:,None]).sum(-1)*self.log_beta.clamp(math.log(2),math.log(8)).exp();return z+offset
    def kl(self):
        result=0
        for kind in ['item','user']:
            mu=getattr(self,kind).weight;ls=getattr(self,kind+'_logstd').weight.clamp(-3,1);
            if hasattr(self,'prior_'+kind):
                pm=getattr(self,'prior_'+kind);pl=getattr(self,'prior_'+kind+'_logstd');result+=.5*(((mu-pm).square()*(-2*pl).exp()).sum()+(self.dim//ls.shape[-1])*((2*(ls-pl)).exp()-1+2*pl-2*ls).sum())
            else:result+=.5*(mu.square().sum()+(self.dim//ls.shape[-1])*(ls.mul(2).exp()-1-2*ls).sum())
        if hasattr(self,'context_matrix'):result+=50*(self.context_matrix.square().sum()+self.context_vector.weight.square().sum()+self.context_outside.weight.square().sum())
        return result+.5*(self.outside.weight[:,0]-self.outside_prior).square().sum()

@torch.no_grad()
def predict(model,d):
    model.eval();allz=[];hs=[]
    for start in range(0,len(d['user']),4096):
        ix=np.arange(start,min(start+4096,len(d['user'])));c=torch.tensor(d['items'][ix]);u=torch.tensor(d['user'][ix]);h=torch.tensor(d['hist'][ix]);mask=(h>0).float();iv=model.vectors('item',c,False);hv=model.vectors('item',h,False)*mask[:,:,None];hm=nn.functional.normalize(hv.sum(1),dim=-1);hc=mask.sum(1,keepdim=True);hm*=torch.sqrt(hc/(hc+10));beta=model.log_beta.clamp(math.log(2),math.log(8)).exp();ctx=torch.tensor(d['context'][ix]);z=model(c,u,False,ctx);hz=model(c,u,False,ctx,hm);allz.append(z.numpy());hs.append(hz.numpy())
    return np.concatenate(allz),np.concatenate(hs)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--final',action='store_true');ap.add_argument('--cache',default='v2/P3_Labs_model_and_code/cache');ap.add_argument('--init');ap.add_argument('--exposure');ap.add_argument('--pool-cache');ap.add_argument('--cut',type=int,default=1737849600);ap.add_argument('--context-model',action='store_true');ap.add_argument('--empirical-prior',action='store_true');ap.add_argument('--warm-density',action='store_true');ap.add_argument('--geometry',choices=['spherical','gaussian'],default='spherical');ap.add_argument('--neg',choices=['user','cluster','global'],default='user');ap.add_argument('--kl',type=float,default=.25);ap.add_argument('--epochs',type=int,default=20);ap.add_argument('--lr',type=float,default=.006);ap.add_argument('--mode',choices=['joint','choice'],default='joint');ap.add_argument('--seed',type=int,default=1027);a=ap.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);torch.set_num_threads(3);torch.manual_seed(a.seed);rng=np.random.default_rng(a.seed);d,e,audit=load_data(a.cache);fi,vi=split_data(d,a.cut);fit=subset(d,fi);val=e if a.final else subset(d,vi);ni=audit['n_items'];nu=audit['n_users'];v=np.load(a.init if a.init else f'v3/states/{a.cut}/p0_u0.5_i0.5_r5.npy')[:,:8].copy();v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);m=interaction_matrix(fit,ni,nu);ic=np.asarray(m.sum(0)).ravel();uc=np.asarray(m.sum(1)).ravel();u=m@v;u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8);s=stats(fit,ni,nu);rate=(s['uh']+6.3)/(s['ui']+10);outside=(logit(np.clip(rate,1e-4,1-1e-4))-np.log(10)-.6).astype(np.float32);model=Variational(v,u,ic,uc,outside,a.geometry,a.context_model);opt=torch.optim.Adam(model.parameters(),lr=a.lr)
    wu,wi,wx=warmup(fit)
    if a.empirical_prior:
        from warm_prior import attach_prior
        attach_prior(model,fit,v,a.cut);wu=wu[:0];wi=wi[:0];wx=wx[:0]
    exposure=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=m.shape).tocsr();tr=np.arange(len(fi)) if a.mode=='joint' else np.flatnonzero(fit['target']<10);N=len(tr)+len(wu);pools=pooled(fit,ni,nu,a.cut,basis=a.exposure,cache=a.pool_cache) if a.neg=='cluster' else None;global_q=s['imp'].astype(float);global_q/=global_q.sum();history=[];best=0
    if not a.final:
        y=binary_labels(val);old=np.load('v4/flex_stack/cv_evidence.npz');lookup={int(k):j for j,k in enumerate(old['indices'])};available=np.array([int(k) in lookup for k in vi]);b=(.75*old['prediction']+.25*old['v3']).reshape(-1,10);base=b[[lookup.get(int(k),0) for k in vi]];pc=base.sum(1,keepdims=True)
    for epoch in range(1,a.epochs+1):
        start=time.time();order=rng.permutation(N);model.train();losses=[]
        for st in range(0,N,4096):
            ix=order[st:st+4096];it=tr[ix[ix<len(tr)]];iw=ix[ix>=len(tr)]-len(tr);sptr=exposure.indptr[wu[iw]];length=np.diff(exposure.indptr)[wu[iw]];neg=exposure.indices[sptr[:,None]+(rng.random((len(iw),9))*length[:,None]).astype(np.int64)];
            if a.neg=='cluster':neg=sample_pool(pools,wu[iw],rng)
            elif a.neg=='global':neg=rng.choice(ni+1,(len(iw),9),p=global_q)
            wc=np.column_stack([wi[iw],neg]);c=torch.tensor(np.concatenate([fit['items'][it],wc]),dtype=torch.long);users=torch.tensor(np.r_[fit['user'][it],wu[iw]],dtype=torch.long);nt=len(it);ctx=torch.tensor(np.r_[fit['context'][it],wx[iw]],dtype=torch.long);z=model(c,users,True,ctx);opt.zero_grad(set_to_none=True)
            if a.mode=='joint':zz=torch.cat([z[:nt],torch.zeros(nt,1)],1)
            else:zz=z[:nt]
            zw=z[nt:]
            if a.warm_density:
                logc=torch.logsumexp(torch.cat([torch.zeros(len(iw),1),zw[:,1:]],1),1,keepdim=True)
                zw=nn.functional.logsigmoid(zw-logc)
            loss=(nn.functional.cross_entropy(zz,torch.tensor(fit['target'][it]),reduction='sum')+nn.functional.cross_entropy(zw,torch.zeros(len(iw),dtype=torch.long),reduction='sum'))/len(ix)+a.kl*model.kl()/N;loss.backward();opt.step();losses.append(loss.item())
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-start,'beta':float(model.log_beta.exp().detach()),'results':[]}
        if a.final:
            if epoch==a.epochs:
                z,hz=predict(model,val);torch.save({'state':model.state_dict(),'args':vars(a),'epoch':epoch},out/'model.pt');np.savez_compressed(out/'evaluation_logits.npz',user=z,history=hz)
            history.append(rec);(out/'metrics.json').write_text(json.dumps({'args':vars(a),'history':history},indent=2));print(json.dumps(rec),flush=True);continue
        z,hz=predict(model,val);winner=(0,None,None)
        for mix in [.5,.75,1]:
            scores=mix*z+(1-mix)*hz
            for event in ['joint','baseline']:
                p=softmax(np.column_stack([scores,np.zeros(len(scores))]),axis=1)[:,:10] if event=='joint' else softmax(scores,axis=1)*pc;auc=roc_auc_score(y[available].ravel(),p[available].ravel());blend=roc_auc_score(y[available].ravel(),(.8*base+.2*p)[available].ravel());rec['results'].append([mix,event,auc,blend])
                if blend>winner[0]:winner=(blend,(mix,event),p)
        if winner[0]>best:best=winner[0];torch.save({'state':model.state_dict(),'args':vars(a),'epoch':epoch},out/'model.pt');np.savez_compressed(out/'logits.npz',user=z,history=hz);np.save(out/'pred.npy',winner[2]);rec['best_config']=winner[1]
        history.append(rec);(out/'metrics.json').write_text(json.dumps({'args':vars(a),'history':history,'best_blend':best},indent=2));print(json.dumps(rec),flush=True)

if __name__=='__main__':main()

