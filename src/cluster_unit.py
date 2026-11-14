"""Regularized choice matrix factorization with warm-up click likelihood."""
import sys,argparse,time,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v2/experiments'))
import torch
from torch import nn
from common import *
from topic import counts
from negative import pooled,sample as sample_pool
from scipy.special import softmax

def representation(fit,ni,nu,dim,folder):
    file=folder/'initial.npz'
    if file.exists():
        z=np.load(file);return z['items'],z['users']
    m=interaction_matrix(fit,ni,nu);ip=np.asarray(m.sum(0)).ravel();up=np.asarray(m.sum(1)).ravel()
    w=sp.diags(1/np.maximum(up,1)**.5)@m@sp.diags(1/np.maximum(ip,1)**.35)
    _,ss,vt=randomized_svd(w,n_components=max(65,dim+1),n_iter=15,random_state=2718)
    v=vt[1:dim+1].T.astype(np.float32);v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);v[0]=0
    u=(m@v)/np.maximum(up[:,None],1)*4
    np.savez_compressed(file,items=v,users=u);return v,u

def warmup(fit):
    keep=(fit['hist']>0)&(fit['hist_ts']<1737331200)
    u=np.broadcast_to(fit['user'][:,None],keep.shape)[keep];i=fit['hist'][keep];t=fit['hist_ts'][keep]
    z=np.rec.fromarrays([u,i,t],names='u,i,t');_,idx=np.unique(z,return_index=True);u=u[idx];i=i[idx];t=t[idx]
    ctx=t%86400//21600+4*((t//86400+3)%7>=5)
    return u,i,ctx

class Latent(nn.Module):
    def __init__(self,v,u,freeze=False,anchor=0,userreg=None,confidence=None,temperature=4.):
        super().__init__();self.item=nn.Embedding.from_pretrained(torch.tensor(v),freeze=freeze,padding_idx=0);self.user=nn.Embedding.from_pretrained(torch.tensor(u),freeze=False,padding_idx=0)
        self.bias=nn.Embedding(len(v),1,padding_idx=0);self.ctx=nn.Embedding(len(v),8,padding_idx=0);nn.init.zeros_(self.bias.weight);nn.init.zeros_(self.ctx.weight)
        self.register_buffer('initial',torch.tensor(v));self.anchor=anchor
        self.register_buffer('userreg',torch.full((len(u),),.01) if userreg is None else torch.tensor(userreg))
        self.register_buffer('confidence',torch.ones(len(u)) if confidence is None else torch.tensor(confidence));self.temperature=temperature
    def forward(self,c,u,ctx,preference=None):
        iv=nn.functional.normalize(self.item(c),dim=-1)
        uv=self.user(u) if preference is None else preference
        uv=self.temperature*nn.functional.normalize(uv,dim=-1)
        if preference is None:uv=uv*self.confidence[u,None]
        score=(iv*uv[:,None]).sum(-1)+self.bias(c).squeeze(-1)+self.ctx(c).gather(2,ctx[:,None,None].expand(-1,c.shape[1],1)).squeeze(-1)
        return score
    def penalty(self,c,u):
        return .10*self.bias(c).square().mean()+.20*self.ctx(c).square().mean()+self.anchor*(self.item(c)-self.initial[c]).square().sum(-1).mean()

@torch.no_grad()
def pred(model,val,pc):
    model.eval();allz=[];histz=[]
    for st in range(0,len(pc),4096):
        dd=subset(val,np.arange(st,min(st+4096,len(pc))));c=torch.tensor(dd['items']);u=torch.tensor(dd['user']);ctx=torch.tensor(dd['context']);h=torch.tensor(dd['hist']);mask=(h>0).float();v=nn.functional.normalize(model.item(h),dim=-1);hp=4*(v*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
        allz.append(model(c,u,ctx).numpy());histz.append(model(c,u,ctx,hp).numpy())
    return np.concatenate(allz),np.concatenate(histz)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--neg',choices=['user','cluster','global'],default='cluster');ap.add_argument('--cache',default='v2/P3_Labs_model_and_code/cache');ap.add_argument('--cut',type=int,default=1737849600);ap.add_argument('--dim',type=int,default=12);ap.add_argument('--epochs',type=int,default=15);ap.add_argument('--lr',type=float,default=.008);ap.add_argument('--freeze',action='store_true');ap.add_argument('--anchor',type=float,default=.1);ap.add_argument('--warm',type=float,default=1);ap.add_argument('--init');ap.add_argument('--exposure');ap.add_argument('--pool-cache');ap.add_argument('--userneg',action='store_true');ap.add_argument('--nobias',action='store_true');ap.add_argument('--bayes',type=float,default=0);ap.add_argument('--final',action='store_true');ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3);torch.manual_seed(817);rng=np.random.default_rng(817)
    cache=a.cache;d,e,audit=load_data(cache);fi,vi=split_data(d,a.cut);fit=subset(d,fi);val=e if a.final else subset(d,vi);ni=audit['n_items'];nu=audit['n_users'];s=stats(fit,ni,nu)
    if a.init:
        v=np.load(a.init)[:,:a.dim].copy();v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-8);m=interaction_matrix(fit,ni,nu);u=4*(m@v)/np.maximum(np.asarray(m.sum(1)),1);np.savez_compressed(out/'initial.npz',items=v,users=u)
    else:v,u=representation(fit,ni,nu,a.dim,out)
    pc=np.full((len(val['user']),1),.63,np.float32) if a.final else (np.load('v2/experiments/n32/valid_pred.npy').sum(1,keepdims=True) if a.cut==1737849600 else np.load('v2/experiments/check_jan25/event_pred.npy').reshape(-1,1))
    y=binary_labels(val) if not a.final else None
    hit=fit['target']<10;tc=fit['items'][hit];tu=fit['user'][hit];tx=fit['context'][hit];ty=fit['target'][hit]
    wu,wi,wx=warmup(fit);q=s['imp'].astype(float);q[0]=0;q/=q.sum();alln=len(tu)+int(len(wu)*a.warm)
    userreg=a.bayes/(np.bincount(np.r_[tu,wu],minlength=nu+1)+5).astype(np.float32) if a.bayes else None
    model=Latent(v,u,a.freeze,a.anchor,userreg)
    model.temperature=4.
    n=np.bincount(np.r_[tu,wu],minlength=nu+1).astype(np.float32);model.register_buffer('confidence',torch.tensor(np.sqrt(n/(n+10))))
    if a.nobias:model.bias.weight.requires_grad_(False);model.ctx.weight.requires_grad_(False)
    exposure=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=(nu+1,ni+1)).tocsr()
    pools=pooled(fit,ni,nu,a.cut,basis=a.exposure,cache=a.pool_cache) if a.neg=='cluster' else None
    opt=torch.optim.Adam(model.parameters(),lr=a.lr);hist=[];best=0
    print('Train clicks',len(tu),'warm-up clicks',len(wu),'dimension',a.dim,flush=True)
    for epoch in range(1,a.epochs+1):
        start=time.time();model.train();order=rng.permutation(alln);total=[]
        for st in range(0,alln,4096):
            ix=order[st:st+4096];iswarm=ix>=len(tu);itr=ix[~iswarm];iw=(ix[iswarm]-len(tu))%len(wu)
            if a.neg=='cluster':negative=sample_pool(pools,wu[iw],rng)
            elif a.neg=='user':
                startp=exposure.indptr[wu[iw]];lengthp=np.diff(exposure.indptr)[wu[iw]];pos=startp[:,None]+(rng.random((len(iw),9))*lengthp[:,None]).astype(np.int64);negative=exposure.indices[pos]
            else:negative=rng.choice(ni+1,(len(iw),9),p=q)
            wc=np.column_stack([wi[iw],negative])
            c=torch.tensor(np.concatenate([tc[itr],wc]),dtype=torch.long);users=torch.tensor(np.r_[tu[itr],wu[iw]],dtype=torch.long);ctx=torch.tensor(np.r_[tx[itr],wx[iw]],dtype=torch.long);target=torch.tensor(np.r_[ty[itr],np.zeros(len(iw),dtype=np.int64)])
            opt.zero_grad(set_to_none=True);z=model(c,users,ctx);loss=nn.functional.cross_entropy(z,target)+model.penalty(c,users);loss.backward();opt.step();total.append(loss.item())
        z,hz=pred(model,val,pc);record={'epoch':epoch,'loss':float(np.mean(total)),'seconds':time.time()-start,'results':[]};winner=(0,None,None)
        for mix in [0,.25,.5,.75,1]:
            p=softmax(mix*z+(1-mix)*hz,axis=1)*pc
            if y is not None:
                auc=roc_auc_score(y.ravel(),p.ravel());record['results'].append([mix,auc])
                if auc>winner[0]:winner=(auc,mix,p)
        if not a.final and winner[0]>best:
            best=winner[0];np.save(out/'valid_pred.npy',winner[2]);np.savez_compressed(out/'valid_logits.npz',user=z,history=hz);torch.save({'state':model.state_dict(),'args':vars(a)},out/'model.pt');record['best_mix']=winner[1]
        if epoch==a.epochs:torch.save({'state':model.state_dict(),'args':vars(a)},out/'last.pt')
        if a.final and epoch==a.epochs:
            torch.save({'state':model.state_dict(),'args':vars(a)},out/'model.pt');np.savez_compressed(out/'evaluation_logits.npz',user=z,history=hz)
        hist.append(record);(out/'metrics.json').write_text(json.dumps({'args':vars(a),'history':hist,'best':best},indent=2));print(json.dumps(record),flush=True)

if __name__=='__main__':main()
