"""Regularized item-specific context residuals with a matched nuisance control."""
import argparse,importlib.util,json,math,time
from pathlib import Path
import numpy as np,torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v148'
spec=importlib.util.spec_from_file_location('v141_support',ROOT/'experiments/v141/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);support=m.support
POOLS=['runtime/v129/pool_parity/pool.npz','runtime/v129/cohort2/pool.npz','runtime/v129/cohort3/pool.npz','runtime/v129/cohort4/pool.npz']

class Model(torch.nn.Module):
    def __init__(self,ni,k,precision):
        super().__init__();self.b=torch.nn.Parameter(torch.zeros(ni));self.warm=torch.nn.Parameter(torch.zeros(ni));self.theta=torch.nn.Parameter(torch.zeros(ni,k)) if k else None;self.k=k;self.precision=precision
    def delta(self,c,ctx,source):
        z=self.b[c]+source[:,None]*self.warm[c]
        if self.k:z=z+self.theta[c,ctx[:,None]%self.k]-self.theta.mean(1)[c]
        return z
    def penalty(self):return .5*(20*(self.b.square().sum()+self.warm.square().sum())+(self.precision*self.theta.square().sum() if self.k else 0))

def fit(cohort):
    start=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];factor=m.arrays(s);catalog=torch.tensor(factor['catalog']);user=torch.tensor(factor['user']);bias=torch.tensor(factor['bias']);d=np.load(ROOT/'runtime/v11/data/train.npz');past=d['timestamp']<cut;f={k:v[past] for k,v in d.items()};del d;mask=(f['hist']>0)&(f['hist_ts']<1737331200);wu=np.broadcast_to(f['user'][:,None],mask.shape)[mask];wi=f['hist'][mask];wt=f['hist_ts'][mask];_,ix=np.unique(np.rec.fromarrays([wu,wi,wt],names='u,i,t'),return_index=True);wu=wu[ix];wi=wi[ix];wc=m.context(wt[ix]);real=f['target']<10;ri=f['items'][real];ru=f['user'][real];rc=f['context'][real];rt=f['target'][real];nr=len(ri);N=nr+len(wu);pool=dict(np.load(ROOT/POOLS[cohort]));configs=[('control',0,0)]+[(f'k{k}_p{p}',k,p) for k in [4,8] for p in [8,32]]
    if cohort>0:
        selected=json.loads((OUT/'selection.json').read_text())['model'];configs=[x for x in configs if x[0] in ['control',selected]]
    nets=[Model(len(catalog),k,p) for _,k,p in configs];opts=[torch.optim.Adam(net.parameters(),lr=.02) for net in nets];rng=np.random.default_rng(148026);records=[]
    for ep in range(8):
        began=time.time();order=rng.permutation(N);totals=np.zeros(len(nets));lr=.004+.016*.5*(1+math.cos(math.pi*ep/7))
        for opt in opts:opt.param_groups[0]['lr']=lr
        for st in range(0,N,2048):
            j=order[st:st+2048];rr=j[j<nr];wa=j[j>=nr]-nr;c=torch.tensor(np.r_[ri[rr],np.c_[wi[wa],support.sample_pool(pool,wu[wa],rng,9)]],dtype=torch.long);u=torch.tensor(np.r_[ru[rr],wu[wa]],dtype=torch.long);ctx=torch.tensor(np.r_[rc[rr],wc[wa]],dtype=torch.long);target=torch.tensor(np.r_[rt[rr],np.zeros(len(wa),int)],dtype=torch.long);source=torch.tensor(np.r_[np.zeros(len(rr)),np.ones(len(wa))],dtype=torch.float32);offset=(catalog[c]*user[u,None]).sum(-1)+bias[c]
            for arm,(net,opt) in enumerate(zip(nets,opts)):
                z=offset+net.delta(c,ctx,source);loss=torch.nn.functional.cross_entropy(z,target)+net.penalty()/N;opt.zero_grad(set_to_none=True);loss.backward();opt.step();totals[arm]+=float(loss.detach())*len(j)
        row=dict(epoch=ep+1,loss={cfg[0]:float(totals[k]/N) for k,cfg in enumerate(configs)},seconds=time.time()-began);records.append(row);print(row,flush=True)
    for (name,k,p),net in zip(configs,nets):torch.save(dict(state=net.state_dict(),contexts=k,precision=p,cut=cut),folder/(name+'.pt'))
    m.write(folder/'fit.json',dict(cohort=cohort,cutoff=cut,real_clicked_events=nr,warm_clicks=len(wu),maximum_training_timestamp=int(f['timestamp'].max()),maximum_warm_timestamp=int(wt[ix].max()),models=[x[0] for x in configs],records=records,query_outcomes_used=False,source_sha256=m.sha(Path(__file__)),plan_sha256=m.sha(ROOT/'experiments/v148/plan.json'),seconds=time.time()-start))

@torch.no_grad()
def predict(cohort):
    folder=OUT/f'cohort{cohort}';d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(d['user']))
    else:
        idx=np.load(ROOT/'runtime/v142/secondary_evidence.npz')['indices'];idx=idx[(d['timestamp'][idx]>=cut)&(d['timestamp'][idx]<cut+86400)]
    c=torch.tensor(d['items'][idx],dtype=torch.long);ctx=torch.tensor(d['context'][idx],dtype=torch.long);control=torch.load(folder/'control.pt',weights_only=False,map_location='cpu')['state']['b'];names=json.loads((folder/'fit.json').read_text())['models'];values={}
    for name in names:
        if name=='control':continue
        ck=torch.load(folder/(name+'.pt'),weights_only=False,map_location='cpu');net=Model(len(control),ck['contexts'],ck['precision']);net.load_state_dict(ck['state']);values[name]=(net.delta(c,ctx,torch.zeros(len(c)))-control[c]).numpy().astype(float)
    np.savez_compressed(folder/'predictions.npz',indices=idx,**values)

def transfer(base,raw,strength):
    b=base.reshape(-1,10);d=.3*np.tanh(raw/.3);return (softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)).ravel()

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for name in v.files:
        if name=='indices':continue
        for strength in [.25,.5,1.]:
            p=transfer(base,v[name],strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(model=name,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{name}_{strength}']=p;print(row,flush=True)
    m.write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
