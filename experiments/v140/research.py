"""Catalog directional uncertainty from historical item Fisher curvature."""
import argparse,json,math,sys,time,importlib.util
from pathlib import Path
import numpy as np,torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v140'
spec=importlib.util.spec_from_file_location('v138_support',ROOT/'experiments/v138/research.py');support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
sys.path.insert(0,str(ROOT/'src'));from uncertainty import torch_direction
write=support.write;sha=support.sha

@torch.no_grad()
def build(cohort):
    began=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];ck=torch.load(ROOT/support.FACTORS[cohort],map_location='cpu',weights_only=False);assert ck['cut']==cut;s=ck['state'];data=dict(np.load(ROOT/'runtime/v11/data/train.npz'));idx=np.flatnonzero(data['timestamp']<cut);f={k:v[idx] for k,v in data.items()};del data;wu,wi=support.warmup(f);pool=dict(np.load(ROOT/support.POOLS[cohort]));rng=np.random.default_rng(140026);mu=s['item.weight'];norm=mu.norm(dim=1).clamp_min(1e-6);iv=mu/norm[:,None];iv[mu.norm(dim=1)<1e-6]=torch.tensor([1.,0.,0.,0.,0.,0.,0.,0.]);uv=torch_direction(s['user.weight'],s['user_logstd.weight'].clamp(-3,1).exp());h=.25*torch.eye(8).repeat(len(mu),1,1);beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()
    for phase,n in [('real',len(idx)),('warm',len(wu))]:
        for st in range(0,n,2048):
            sl=slice(st,st+2048)
            if phase=='real':c=torch.tensor(f['items'][sl],dtype=torch.long);u=torch.tensor(f['user'][sl],dtype=torch.long)
            else:c=torch.tensor(np.c_[wi[sl],support.sample_pool(pool,wu[sl],rng,9)],dtype=torch.long);u=torch.tensor(wu[sl],dtype=torch.long)
            dot=(iv[c]*uv[u,None]).sum(-1);z=beta*dot+eb*(s['exposure_items'][c]*s['exposure_user.weight'][u,None]/math.sqrt(8)).sum(-1)+s['item_bias.weight'][c,0]+s['outside.weight'][u]
            p=torch.softmax(torch.cat([z,torch.zeros(len(u),1)],1),1)[:,:10] if phase=='real' else torch.softmax(z,1)
            grad=beta*(uv[u,None]-dot[:,:,None]*iv[c])/norm[c,None];weight=p*(1-p)*(1 if phase=='real' else 2)
            block=weight[:,:,None,None]*grad[:,:,:,None]*grad[:,:,None,:];h.index_add_(0,c.ravel(),block.reshape(-1,8,8))
        print(dict(stage='fisher',cohort=cohort,phase=phase,events=n),flush=True)
    covariance=torch.empty_like(h);min_eigen=1.;trace_error=0.
    for st in range(0,len(h),2048):
        v=iv[st:st+2048];radial=v[:,:,None]*v[:,None,:];proj=torch.eye(8)[None]-radial;inv=torch.linalg.inv(h[st:st+2048]);t=proj@inv@proj;t=t*7/t.diagonal(dim1=1,dim2=2).sum(1)[:,None,None].clamp_min(1e-8);a=t+radial;a=(a+a.transpose(1,2))/2;covariance[st:st+2048]=a;min_eigen=min(min_eigen,float(torch.linalg.eigvalsh(a).min()));trace_error=max(trace_error,float((a.diagonal(dim1=1,dim2=2).sum(1)-8).abs().max()))
    assert min_eigen>0 and trace_error<1e-4,(min_eigen,trace_error);torch.linalg.cholesky(covariance);np.savez_compressed(folder/'covariance.npz',covariance=covariance.numpy());write(folder/'fit.json',dict(cohort=cohort,cutoff=cut,real_events=len(idx),warm_events=len(wu),min_eigenvalue=min_eigen,trace8_max_error=trace_error,maximum_training_timestamp=int(f['timestamp'].max()),query_outcomes_used=False,factor_sha256=sha(ROOT/support.FACTORS[cohort]),source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'experiments/v140/plan.json'),seconds=time.time()-began))

@torch.no_grad()
def predict(cohort):
    began=time.time();folder=OUT/f'cohort{cohort}';s=torch.load(ROOT/support.FACTORS[cohort],map_location='cpu',weights_only=False)['state'];cov=torch.tensor(np.load(folder/'covariance.npz')['covariance']);chols={str(v):torch.linalg.cholesky(v*cov+(1-v)*torch.eye(8)[None]) for v in [.5,1.]};data=dict(np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz')));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v136/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(data['user']))
    else:
        idx=np.load(ROOT/'runtime/v136/secondary_evidence.npz')['indices'];idx=idx[(data['timestamp'][idx]>=cut)&(data['timestamp'][idx]<cut+86400)]
    eps=support.noise();itemeps=eps[:,8:88].reshape(len(eps),10,8);parts={k:[] for k in ['original',*chols]};beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()
    for st in range(0,len(idx),128):
        ix=idx[st:st+128];c=torch.tensor(data['items'][ix],dtype=torch.long);u=torch.tensor(data['user'][ix],dtype=torch.long);uv=torch.nn.functional.normalize(s['user.weight'][u,None]+s['user_logstd.weight'][u,None].clamp(-3,1).exp()*eps[None,:,:8],dim=-1);ex=(s['exposure_user.weight'][u,None]+s['exposure_logstd.weight'][u,None].clamp(-4,1).exp()*eps[None,:,88:])/math.sqrt(8);offset=torch.einsum('nkd,nsd->nsk',s['exposure_items'][c],ex)*eb+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]
        for name in parts:
            ei=itemeps[None] if name=='original' else torch.einsum('nkde,ske->nskd',chols[name][c],itemeps)
            iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*ei,dim=-1);z=(iv*uv[:,:,None]).sum(-1)*beta+offset;p=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2).mean(1);parts[name].append(p.numpy())
    values={k:np.concatenate(v).astype(float) for k,v in parts.items()};np.savez_compressed(folder/'predictions.npz',indices=idx,**values);write(folder/'prediction_audit.json',dict(queries=len(idx),seconds=time.time()-began,maximum_probability_sum_error=max(float(abs(v.sum(1)-1).max()) for v in values.values())));print(dict(stage='predicted',cohort=cohort,seconds=time.time()-began),flush=True)

def transfer(base,original,alternative,mode,strength):
    b=base.reshape(-1,10)
    if mode=='choice':d=np.log(alternative[:,:10]/original[:,:10]);d-=d.mean(1,keepdims=True);d=.3*np.tanh(d/.3);p=softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)
    else:d=np.log(alternative[:,:10]/alternative[:,10:])-np.log(original[:,:10]/original[:,10:]);d=.3*np.tanh(d/.3);p=softmax(np.c_[np.log(b/(1-b.sum(1,keepdims=True)))+strength*d,np.zeros(len(b))],axis=1)[:,:10]
    return p.ravel()

def screen():
    e=np.load(ROOT/'runtime/v136/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');assert np.array_equal(e['indices'],v['indices']);base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};rows=[];pred={}
    for shrink in ['0.5','1.0']:
        for mode in ['choice','joint']:
            for strength in [.5,1.]:
                p=transfer(base,v['original'],v[shrink],mode,strength);g={k:float(roc_auc_score(y[m],p[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(shrinkage=float(shrink),mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{shrink}_{mode}_{strength}']=p;print(row,flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':build(a.cohort);predict(a.cohort)
    else:screen()
