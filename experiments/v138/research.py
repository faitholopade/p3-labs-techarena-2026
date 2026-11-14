"""Matched continuation with spherical rather than ambient Gaussian entropy."""
import argparse,hashlib,json,math,time,sys
from pathlib import Path
import numpy as np,torch,scipy.sparse as sp
from scipy.special import roots_hermitenorm,roots_genlaguerre,softmax
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v138'
sys.path.insert(0,str(ROOT/'experiments/v125'))
from research import warmup,sample_pool,log_radial_integral,noise
CUTS=[1737676800,1737763200,1737849600,1737936000]
FACTORS=['runtime/benchmark/inputs/factors/friday.pt','runtime/v17/validation/factors/earlier_factor_users_only.pt','runtime/v17/validation/factors/later_factor_users_only.pt','runtime/v17/models/factor_users_only.pt']
POOLS=['runtime/v124/pools/exposure_pools_1737676800_64.npz','runtime/v129/cohort2/pool.npz','runtime/v129/cohort3/pool.npz','runtime/v129/cohort4/pool.npz']
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')

def angular_curve(logk,n):
    z,wz=roots_hermitenorm(n);t,wt=roots_genlaguerre(n,2.5);w=wz[:,None]*wt[None,:]/math.sqrt(2*math.pi)/math.gamma(3.5);out=[]
    for lk in logk:
        k=math.exp(lk);a=k*(k+z[:,None])/np.sqrt((k+z[:,None])**2+2*t[None,:]);value=log_radial_integral(a)-.5*k*k-math.log(48.);out.append(float((w*value).sum()))
    return np.array(out)

def curve():
    path=OUT/'angular_kl.npz'
    if path.exists():return dict(np.load(path))
    began=time.time();grid=np.linspace(-8,6,1025);value=angular_curve(grid,64);check=angular_curve(grid[::8],96);err=float(abs(value[::8]-check).max());assert err<.003,err;assert value.min()>-1e-6 and np.diff(value).min()>-1e-7;value=np.maximum(value,0)
    # For a tightly concentrated direction in dimension8, the KL slope tends to7.
    slope=float((value[-1]-value[-2])/(grid[-1]-grid[-2]));assert 6.95<slope<7.05
    np.savez_compressed(path,logk=grid,value=value);write(OUT/'angular_kl_verification.json',dict(quadrature64_vs96_max_error=err,high_concentration_slope=slope,minimum_kl=float(value.min()),points=len(grid),seconds=time.time()-began));print(dict(stage='angular_curve',error=err,seconds=time.time()-began),flush=True);return dict(logk=grid,value=value)

class Factors(torch.nn.Module):
    def __init__(self,s,lookup):
        super().__init__();fixed={'initial','outside_prior','exposure_items'};self.fixed={k:v.clone() for k,v in s.items() if k in fixed};self.p=torch.nn.ParameterDict({k.replace('.','__'):torch.nn.Parameter(v.clone()) for k,v in s.items() if k not in fixed});self.table=torch.tensor(lookup['value'],dtype=torch.float32);self.lower=float(lookup['logk'][0]);self.step=float(lookup['logk'][1]-lookup['logk'][0])
    def get(self,k):return self.fixed[k] if k in self.fixed else self.p[k.replace('.','__')]
    def state(self):return {**self.fixed,**{k.replace('__','.'):v.detach().clone() for k,v in self.p.items()}}
    def forward(self,c,u,ei,eu,ee,sign):
        iv=torch.nn.functional.normalize(self.get('item.weight')[c]+sign*self.get('item_logstd.weight')[c].clamp(-3,1).exp()*ei,dim=-1);uv=torch.nn.functional.normalize(self.get('user.weight')[u]+sign*self.get('user_logstd.weight')[u].clamp(-3,1).exp()*eu,dim=-1);ex=(self.get('exposure_user.weight')[u]+sign*self.get('exposure_logstd.weight')[u].clamp(-4,1).exp()*ee)/math.sqrt(8);beta=self.get('log_beta').clamp(math.log(2),math.log(8)).exp();eb=self.get('log_exposure_beta').clamp(math.log(.5),math.log(8)).exp();return (iv*uv[:,None]).sum(-1)*beta+(self.get('exposure_items')[c]*ex[:,None]).sum(-1)*eb+self.get('item_bias.weight')[c,0]+self.get('outside.weight')[u]
    def kl(self,angular):
        value=0
        for kind in ['item','user']:
            mu=self.get(kind+'.weight');ls=self.get(kind+'_logstd.weight').clamp(-3,1)
            if angular:
                lk=.5*torch.log(mu.square().sum(1).clamp_min(1e-16))-ls[:,0];x=((lk-self.lower)/self.step).clamp(0,len(self.table)-1.00001);i=x.floor().long();value+=((1-(x-i))*self.table[i]+(x-i)*self.table[i+1]).sum()
            else:value+=.5*(mu.square().sum()+8*(ls.mul(2).exp()-1-2*ls).sum())
        return value+.5*(self.get('outside.weight')[:,0]-self.get('outside_prior')).square().sum()

def fit(cohort):
    lookup=curve();start=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=CUTS[cohort];ck=torch.load(ROOT/FACTORS[cohort],map_location='cpu',weights_only=False);assert ck['cut']==cut;s=ck['state'];data=dict(np.load(ROOT/'runtime/v11/data/train.npz'));idx=np.flatnonzero(data['timestamp']<cut);f={k:v[idx] for k,v in data.items()};del data;wu,wi=warmup(f);nr=len(idx);N=nr+len(wu);pool=dict(np.load(ROOT/POOLS[cohort]));nu=len(s['user.weight']);ni=len(s['item.weight']);counts=np.bincount(f['user'],minlength=nu);imp=sp.coo_matrix((np.ones(nr*10,np.float32),(np.repeat(f['user'],10),f['items'].ravel())),shape=(nu,ni)).tocsr();profile=imp@s['exposure_items'].numpy()/np.maximum(10*counts[:,None],1);dense=counts>=10;ridge=Ridge(alpha=10).fit(profile[dense],s['exposure_user.weight'].numpy()[dense]);prior=torch.tensor(ridge.predict(profile),dtype=torch.float32);np.savez_compressed(folder/'prior.npz',mean=prior.numpy(),coef=ridge.coef_,intercept=ridge.intercept_);del imp,profile;nets=[Factors(s,lookup),Factors(s,lookup)];opts=[torch.optim.Adam(n.parameters(),lr=.001) for n in nets];rng=np.random.default_rng(138026);torch.manual_seed(138026);records=[]
    for ep in range(6):
        began=time.time();order=rng.permutation(N);totals=[0.,0.];lr=.0002+.0008*.5*(1+math.cos(math.pi*ep/5))
        for opt in opts:opt.param_groups[0]['lr']=lr
        for st in range(0,N,2048):
            j=order[st:st+2048];ri=j[j<nr];wa=j[j>=nr]-nr;nt=len(ri);nw=len(wa);c=torch.tensor(np.concatenate([f['items'][ri],np.c_[wi[wa],sample_pool(pool,wu[wa],rng,9)]]),dtype=torch.long);u=torch.tensor(np.r_[f['user'][ri],wu[wa]],dtype=torch.long);target=torch.tensor(f['target'][ri],dtype=torch.long);ei=torch.randn(len(j),10,8);eu=torch.randn(len(j),8);ee=torch.randn(len(j),8);labels=(torch.arange(10)[None]==target[:,None]).ravel();npos=int(labels.sum());nneg=len(labels)-npos;pairs=torch.randint(nneg,(npos,32)) if npos else None
            for arm,(net,opt) in enumerate(zip(nets,opts)):
                draws=[net(c,u,ei,eu,ee,sign) for sign in [1,-1]];real=sum(torch.nn.functional.cross_entropy(torch.cat([z[:nt],torch.zeros(nt,1)],1),target,reduction='sum') for z in draws)/2;warm=sum(torch.nn.functional.cross_entropy(z[nt:],torch.zeros(nw,dtype=torch.long),reduction='sum') for z in draws)/2;mu=net.get('exposure_user.weight');ls=net.get('exposure_logstd.weight').clamp(-4,1);extra=.5*((mu-prior).square().sum()/.5+8*((2*ls).exp()/.5-1+math.log(.5)-2*ls).sum())+10*net.get('item_bias.weight').square().sum();loss=(real+2*warm)/len(j)+(.25*net.kl(arm==1)+.5*extra)/N
                if npos:
                    prob=torch.stack([torch.softmax(torch.cat([z[:nt],torch.zeros(nt,1)],1),1)[:,:10] for z in draws]).mean(0).ravel().clamp(1e-6,1-1e-6);l=torch.logit(prob);loss+=.5*(nt/len(j))*torch.nn.functional.softplus(l[~labels][pairs]-l[labels,None]).mean()
                opt.zero_grad(set_to_none=True);loss.backward();opt.step();totals[arm]+=float(loss.detach())*len(j)
        rec=dict(epoch=ep+1,lr=lr,control_loss=totals[0]/N,angular_loss=totals[1]/N,seconds=time.time()-began);records.append(rec);print(rec,flush=True)
    for name,net in zip(['control','angular'],nets):torch.save(dict(state=net.state(),cut=cut,epoch=6),folder/f'{name}.pt')
    report=dict(cohort=cohort,cutoff=cut,training_events=nr,warm_clicks=len(wu),maximum_training_timestamp=int(f['timestamp'].max()),query_outcomes_used=False,shared_randomness=True,records=records,seconds=time.time()-start,factor_sha256=sha(ROOT/FACTORS[cohort]),source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'experiments/v138/plan.json'));write(folder/'fit.json',report)

@torch.no_grad()
def predict(cohort):
    folder=OUT/f'cohort{cohort}';states=[torch.load(folder/f'{name}.pt',map_location='cpu',weights_only=False)['state'] for name in ['control','angular']];data=dict(np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz')));cut=CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v136/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(data['user']))
    else:
        idx=np.load(ROOT/'runtime/v136/secondary_evidence.npz')['indices'];idx=idx[(data['timestamp'][idx]>=cut)&(data['timestamp'][idx]<cut+86400)]
    eps=noise();parts=[[],[]]
    for st in range(0,len(idx),128):
        ix=idx[st:st+128];c=torch.tensor(data['items'][ix],dtype=torch.long);u=torch.tensor(data['user'][ix],dtype=torch.long)
        for k,s in enumerate(states):
            iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*eps[None,:,8:88].reshape(1,len(eps),10,8),dim=-1);uv=torch.nn.functional.normalize(s['user.weight'][u,None]+s['user_logstd.weight'][u,None].clamp(-3,1).exp()*eps[None,:,:8],dim=-1);ex=(s['exposure_user.weight'][u,None]+s['exposure_logstd.weight'][u,None].clamp(-4,1).exp()*eps[None,:,88:])/math.sqrt(8);beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp();z=(iv*uv[:,:,None]).sum(-1)*beta+torch.einsum('nkd,nsd->nsk',s['exposure_items'][c],ex)*eb+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None];p=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2).mean(1).numpy();parts[k].append(p)
    np.savez_compressed(folder/'predictions.npz',indices=idx,control=np.concatenate(parts[0]).astype(float),angular=np.concatenate(parts[1]).astype(float));print(dict(stage='predicted',cohort=cohort,queries=len(idx)),flush=True)

def screen():
    e=np.load(ROOT/'runtime/v136/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];b=base.reshape(-1,10);ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};rows=[];a=v['angular'];c=v['control'];joint=np.log(a[:,:10]/a[:,10:])-np.log(c[:,:10]/c[:,10:]);choice=np.log(a[:,:10])-np.log(c[:,:10]);choice-=choice.mean(1,keepdims=True)
    for mode,delta in [('choice',choice),('joint',joint)]:
        delta=.3*np.tanh(delta/.3)
        for strength in [.25,.5,1.]:
            if mode=='choice':p=softmax(np.log(b)+strength*delta,axis=1)*b.sum(1,keepdims=True)
            else:
                z=np.log(b/(1-b.sum(1,keepdims=True)))+strength*delta;p=softmax(np.c_[z,np.zeros(len(z))],axis=1)[:,:10]
            p=p.ravel();g={k:float(roc_auc_score(y[m],p[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));r=dict(mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(r);print(json.dumps(r),flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba,scope='Reused V136 proxy; exact V136 heldout head unavailable'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen','curve'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3);OUT.mkdir(exist_ok=True,parents=True)
    if a.stage=='curve':curve()
    elif a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
