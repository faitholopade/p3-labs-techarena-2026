"""Exposure-likelihood catalog refinement from pre-cutoff candidate displays."""
import argparse,hashlib,json,math,sys,time,importlib.util
from pathlib import Path
import numpy as np,scipy.sparse as sp,torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v143'
spec=importlib.util.spec_from_file_location('v138_support',ROOT/'experiments/v138/research.py');support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
write=support.write;sha=support.sha

class Exposure(torch.nn.Module):
    def __init__(self,q,v,b,a):
        super().__init__();self.q=torch.nn.Parameter(q.clone());self.v=torch.nn.Parameter(v.clone());self.b=torch.nn.Parameter(b.clone());self.a=torch.nn.Parameter(a.clone())
    def catalog(self):return torch.nn.functional.normalize(self.q,dim=1)
    def forward(self,c,u,lognoise):return (self.catalog()[c]*self.v[u,None]).sum(-1)+self.b[c]+self.a[u,None]-math.log(4)-lognoise[c]

def fit(cohort):
    began=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];ck=torch.load(ROOT/support.FACTORS[cohort],map_location='cpu',weights_only=False);assert ck['cut']==cut;s=ck['state'];d=np.load(ROOT/'runtime/v11/data/train.npz');idx=np.flatnonzero(d['timestamp']<cut);items=d['items'][idx];users=d['user'][idx];q=s['exposure_items'];nu=len(s['user.weight']);ni=len(q);n=len(idx);M=n*10;counts=np.bincount(users,minlength=nu);pop=np.bincount(items.ravel(),minlength=ni).astype(float)+1;pop[0]=0;pop/=pop.sum();cdf=np.cumsum(pop);lognoise=torch.tensor(np.log(np.maximum(pop,1e-30)),dtype=torch.float32);imp=sp.coo_matrix((np.ones(M,np.float32),(np.repeat(users,10),items.ravel())),shape=(nu,ni)).tocsr();profile=imp@q.numpy()/np.maximum(10*counts[:,None],1);mean=pop@q.numpy();center=q.numpy()-mean;cov=(center.T*pop)@center;v=(profile-mean)@np.linalg.inv(cov+.02*np.eye(8));v*=counts[:,None]/(counts[:,None]+5);v*=np.minimum(1,6/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-9));a=-v@mean-.5*np.einsum('nd,de,ne->n',v,cov,v);v=torch.tensor(v,dtype=torch.float32);a=torch.tensor(a,dtype=torch.float32);b=lognoise.clone();del imp,profile;anchors=[20,80];nets=[Exposure(q,v,b,a) for _ in anchors];opts=[torch.optim.Adam(net.parameters(),lr=.02) for net in nets];rng=np.random.default_rng(143026);records=[]
    for epoch in range(6):
        start=time.time();order=rng.permutation(n);totals=[0.,0.];lr=.004+.016*.5*(1+math.cos(math.pi*epoch/5))
        for opt in opts:opt.param_groups[0]['lr']=lr
        for st in range(0,n,2048):
            rows=order[st:st+2048];u=torch.tensor(users[rows],dtype=torch.long);negative=np.searchsorted(cdf,rng.random((len(rows),40)));c=torch.tensor(np.c_[items[rows],negative],dtype=torch.long)
            for j,(anchor,net,opt) in enumerate(zip(anchors,nets,opts)):
                z=net(c,u,lognoise);nce=(torch.nn.functional.softplus(-z[:,:10]).sum()+torch.nn.functional.softplus(z[:,10:]).sum())/(len(rows)*10);penalty=.5*(anchor*(net.q-q).square().sum()+net.v.square().sum()+20*(net.b-b).square().sum()+(net.a-a).square().sum())/M;loss=nce+penalty;opt.zero_grad(set_to_none=True);loss.backward();opt.step();totals[j]+=float(loss.detach())*len(rows)
        row=dict(epoch=epoch+1,learning_rate=lr,losses={str(k):totals[j]/n for j,k in enumerate(anchors)},seconds=time.time()-start);records.append(row);print(row,flush=True)
    movement={}
    for anchor,net in zip(anchors,nets):
        catalog=net.catalog().detach();catalog[0]=0;torch.save(dict(catalog=catalog,display_user=net.v.detach(),display_item_bias=net.b.detach(),display_user_intercept=net.a.detach(),cut=cut,anchor=anchor),folder/f'anchor{anchor}.pt');movement[str(anchor)]=dict(mean_distance=float((catalog[1:]-q[1:]).norm(dim=1).mean()),mean_cosine=float((catalog[1:]*q[1:]).sum(1).mean()))
    write(folder/'fit.json',dict(cutoff=cut,events=n,positive_exposures=M,maximum_training_timestamp=int(d['timestamp'][idx].max()),query_exposures_used=False,query_outcomes_used=False,records=records,movement=movement,seconds=time.time()-began,factor_sha256=sha(ROOT/support.FACTORS[cohort]),source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'experiments/v143/plan.json')))

@torch.no_grad()
def predict(cohort):
    folder=OUT/f'cohort{cohort}';s=torch.load(ROOT/support.FACTORS[cohort],map_location='cpu',weights_only=False)['state'];catalog={'original':s['exposure_items'],**{str(a):torch.load(folder/f'anchor{a}.pt',map_location='cpu',weights_only=False)['catalog'] for a in [20,80]}};d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(d['user']))
    else:
        idx=np.load(ROOT/'runtime/v142/secondary_evidence.npz')['indices'];idx=idx[(d['timestamp'][idx]>=cut)&(d['timestamp'][idx]<cut+86400)]
    eps=support.noise();parts={k:[] for k in catalog};beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp();start=time.time()
    for st in range(0,len(idx),256):
        ix=idx[st:st+256];c=torch.tensor(d['items'][ix],dtype=torch.long);u=torch.tensor(d['user'][ix],dtype=torch.long);iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*eps[None,:,8:88].reshape(1,len(eps),10,8),dim=-1);uv=torch.nn.functional.normalize(s['user.weight'][u,None]+s['user_logstd.weight'][u,None].clamp(-3,1).exp()*eps[None,:,:8],dim=-1);ex=(s['exposure_user.weight'][u,None]+s['exposure_logstd.weight'][u,None].clamp(-4,1).exp()*eps[None,:,88:])/math.sqrt(8);offset=(iv*uv[:,:,None]).sum(-1)*beta+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]
        for name,q in catalog.items():
            z=offset+eb*torch.einsum('nkd,nsd->nsk',q[c],ex);p=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2).mean(1).numpy();parts[name].append(p)
    values={k:np.concatenate(v).astype(float) for k,v in parts.items()};np.savez_compressed(folder/'predictions.npz',indices=idx,**values);print(dict(stage='predict',cohort=cohort,queries=len(idx),seconds=time.time()-start),flush=True)

def transfer(base,original,alternative,mode,strength):
    b=base.reshape(-1,10)
    if mode=='choice':d=np.log(alternative[:,:10]/original[:,:10]);d-=d.mean(1,keepdims=True);d=.3*np.tanh(d/.3);p=softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)
    else:d=np.log(alternative[:,:10]/alternative[:,10:])-np.log(original[:,:10]/original[:,10:]);d=.3*np.tanh(d/.3);p=softmax(np.c_[np.log(b/(1-b.sum(1,keepdims=True)))+strength*d,np.zeros(len(b))],axis=1)[:,:10]
    return p.ravel()

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];assert np.array_equal(e['indices'],v['indices']);masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};rows=[];pred={}
    for anchor in [20,80]:
        for mode in ['choice','joint']:
            for strength in [.25,.5,1.]:
                p=transfer(base,v['original'],v[str(anchor)],mode,strength);g={k:float(roc_auc_score(y[m],p[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(anchor=anchor,mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{anchor}_{mode}_{strength}']=p;print(row,flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
