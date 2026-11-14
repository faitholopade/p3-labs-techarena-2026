"""Learn an additional collaborative residual under a matched calibration control."""
import argparse,hashlib,json,math,time,sys
from pathlib import Path
import numpy as np,torch,scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v137'
sys.path.insert(0,str(ROOT/'experiments/v125'))
from research import warmup,sample_pool
CUTS=[1737676800,1737763200,1737849600,1737936000]
FACTORS=['runtime/benchmark/inputs/factors/friday.pt','runtime/v17/validation/factors/earlier_factor_users_only.pt','runtime/v17/validation/factors/later_factor_users_only.pt','runtime/v17/models/factor_users_only.pt']
POOLS=['runtime/v124/pools/exposure_pools_1737676800_64.npz','runtime/v129/cohort2/pool.npz','runtime/v129/cohort3/pool.npz','runtime/v129/cohort4/pool.npz']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

class Offset:
    def __init__(self,s):
        self.s=s;mu=s['item.weight'];sd=s['item_logstd.weight'].clamp(-3,1).exp();self.iv=mu/(mu.square().sum(-1,keepdim=True)+7*sd.square()).sqrt().clamp_min(1e-8);mu=s['user.weight'];sd=s['user_logstd.weight'].clamp(-3,1).exp();self.uv=mu/(mu.square().sum(-1,keepdim=True)+7*sd.square()).sqrt().clamp_min(1e-8);self.beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();self.eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()/math.sqrt(8)
    def __call__(self,c,u):
        s=self.s;return (self.iv[c]*self.uv[u,None]).sum(-1)*self.beta+(s['exposure_items'][c]*s['exposure_user.weight'][u,None]).sum(-1)*self.eb+s['item_bias.weight'][c,0]

class Residual(torch.nn.Module):
    def __init__(self,nu,ni,rank,ui=None,ii=None):
        super().__init__();self.rank=rank;self.logtemp=torch.nn.Parameter(torch.zeros(()));self.bias=torch.nn.Embedding(ni,1);torch.nn.init.zeros_(self.bias.weight)
        if rank:
            self.user=torch.nn.Embedding.from_pretrained(torch.tensor(ui[:,:rank].copy()),freeze=False);self.item=torch.nn.Embedding.from_pretrained(torch.tensor(ii[:,:rank].copy()),freeze=False)
    def forward(self,z,c,u):
        out=z*self.logtemp.clamp(-.3,.3).exp()+self.bias(c).squeeze(-1)
        if self.rank:out=out+(self.item(c)*self.user(u)[:,None]).sum(-1)
        return out
    def penalty(self,total):
        val=20*self.bias.weight.square().sum()/total+.01*self.logtemp.square()
        if self.rank:val+=1.5*(self.user.weight.square().sum()+self.item.weight.square().sum())/total
        return val

def fit(cohort):
    started=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=CUTS[cohort];ck=torch.load(ROOT/FACTORS[cohort],map_location='cpu',weights_only=False);assert ck['cut']==cut;s=ck['state'];data=dict(np.load(ROOT/'runtime/v11/data/train.npz'));ix=np.flatnonzero(data['timestamp']<cut);f={k:v[ix] for k,v in data.items()};del data;pool=dict(np.load(ROOT/POOLS[cohort]));off=Offset(s);nu,ni=len(s['user.weight']),len(s['item.weight']);real=np.flatnonzero(f['target']<10);ru=f['user'][real];rc=f['items'][real];ry=f['target'][real];warmu,warmi=warmup(f);N=len(real)+len(warmu);effective=len(real)+2*len(warmu);zparts=[]
    with torch.no_grad():
        for st in range(0,len(real),8192):zparts.append(off(torch.tensor(rc[st:st+8192],dtype=torch.long),torch.tensor(ru[st:st+8192],dtype=torch.long)).numpy())
    rz=np.concatenate(zparts);weights=-softmax(rz,axis=1);weights[np.arange(len(ry)),ry]+=1;rr=np.repeat(ru,10);cc=rc.ravel();resid=sp.coo_matrix((weights.ravel(),(rr,cc)),shape=(nu,ni)).tocsr();uc=np.bincount(rr,minlength=nu);ic=np.bincount(cc,minlength=ni);resid=sp.diags(1/np.sqrt(uc+10))@resid@sp.diags(1/np.sqrt(ic+10));left,sv,right=randomized_svd(resid,n_components=16,n_iter=7,random_state=137026);ui=(left*np.sqrt(nu)*.1).astype(np.float32);ii=(right.T*np.sqrt(ni)*.1).astype(np.float32);ui[uc==0]=0;ii[ic==0]=0;np.savez_compressed(folder/'initial.npz',user=ui,item=ii,singular=sv);models={};optimizers={};records=[];torch.manual_seed(137026)
    for rank in [0,8,16]:models[rank]=Residual(nu,ni,rank,ui,ii);optimizers[rank]=torch.optim.Adam(models[rank].parameters(),lr=.02)
    rng=np.random.default_rng(137027)
    for ep in range(8):
        start=time.time();order=rng.permutation(N);totals={k:0. for k in models}
        for rank,opt in optimizers.items():opt.param_groups[0]['lr']=.002+.018*.5*(1+math.cos(math.pi*ep/7))
        for st in range(0,N,4096):
            ids=order[st:st+4096];r=ids[ids<len(real)];w=ids[ids>=len(real)]-len(real);neg=sample_pool(pool,warmu[w],rng,9);c=torch.tensor(np.concatenate([rc[r],np.c_[warmi[w],neg]]),dtype=torch.long);u=torch.tensor(np.r_[ru[r],warmu[w]],dtype=torch.long);y=torch.tensor(np.r_[ry[r],np.zeros(len(w),dtype=np.int64)],dtype=torch.long);ww=torch.tensor(np.r_[np.ones(len(r)),np.full(len(w),2)],dtype=torch.float32)
            with torch.no_grad():z=off(c,u)
            for rank,net in models.items():
                opt=optimizers[rank];p=net(z,c,u);loss=(torch.nn.functional.cross_entropy(p,y,reduction='none')*ww).sum()/len(ids)+net.penalty(effective);opt.zero_grad(set_to_none=True);loss.backward();opt.step();totals[rank]+=float(loss.detach())*len(ids)
        rec=dict(epoch=ep+1,loss={str(k):v/N for k,v in totals.items()},seconds=time.time()-start);records.append(rec);print(rec,flush=True)
    for rank,net in models.items():torch.save(dict(state=net.state_dict(),rank=rank,cutoff=cut,nu=nu,ni=ni),folder/f'rank{rank}.pt')
    report=dict(cohort=cohort,cutoff=cut,real_clicked_events=len(real),warm_clicks=len(warmu),max_training_timestamp=int(f['timestamp'].max()),factor_sha256=sha(ROOT/FACTORS[cohort]),pool_sha256=sha(ROOT/POOLS[cohort]),records=records,seconds=time.time()-started,plan_sha256=sha(ROOT/'experiments/v137/plan.json'));write(folder/'fit.json',report)

@torch.no_grad()
def predict(cohort):
    folder=OUT/f'cohort{cohort}';s=torch.load(ROOT/FACTORS[cohort],map_location='cpu',weights_only=False)['state'];off=Offset(s);data=dict(np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz')));cut=CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v136/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(data['user']))
    else:
        idx=np.load(ROOT/'runtime/v136/secondary_evidence.npz')['indices'];idx=idx[(data['timestamp'][idx]>=cut)&(data['timestamp'][idx]<cut+86400)]
    nets={};parts={k:[] for k in [0,8,16]}
    for rank in parts:
        ck=torch.load(folder/f'rank{rank}.pt',map_location='cpu',weights_only=False);net=Residual(ck['nu'],ck['ni'],rank,np.zeros((ck['nu'],rank),np.float32),np.zeros((ck['ni'],rank),np.float32));net.load_state_dict(ck['state']);nets[rank]=net.eval()
    for st in range(0,len(idx),2048):
        ii=idx[st:st+2048];c=torch.tensor(data['items'][ii],dtype=torch.long);u=torch.tensor(data['user'][ii],dtype=torch.long);z=off(c,u)
        for rank,net in nets.items():parts[rank].append(net(z,c,u).numpy())
    values={f'rank{k}':np.concatenate(v) for k,v in parts.items()};np.savez_compressed(folder/'predictions.npz',indices=idx,**values);print(dict(stage='predicted',cohort=cohort,queries=len(idx)),flush=True)

def screen():
    e=np.load(ROOT/'runtime/v136/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');assert np.array_equal(e['indices'],v['indices']);base=e['candidate'];y=e['labels'];b=base.reshape(-1,10);d=np.load(ROOT/'runtime/v11/data/train.npz');ts=d['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};rows=[]
    for rank in [8,16]:
        delta=v[f'rank{rank}'].astype(float)-v['rank0'];delta-=delta.mean(1,keepdims=True);delta=.3*np.tanh(delta/.3)
        for strength in [.25,.5,1.]:
            pred=(softmax(np.log(b)+strength*delta,axis=1)*b.sum(1,keepdims=True)).ravel();g={k:float(roc_auc_score(y[m],pred[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,pred)-log_loss(y,base));r=dict(rank=rank,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(r);print(json.dumps(r),flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba,scope='V136 proxy, reused development; exact V136 held-out checkpoint unavailable'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cohort',type=int);p.add_argument('--stage',choices=['run','screen'],required=True);a=p.parse_args();torch.set_num_threads(3);OUT.mkdir(exist_ok=True,parents=True)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
