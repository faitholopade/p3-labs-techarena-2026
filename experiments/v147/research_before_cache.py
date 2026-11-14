"""Correlate uncertain click propensity with preference/exposure latents."""
import argparse,importlib.util,json,math,time,sys
from pathlib import Path
import numpy as np,torch
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v147'
spec=importlib.util.spec_from_file_location('v143_support',ROOT/'experiments/v143/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);support=m.support
sys.path.insert(0,str(ROOT/'src'));from uncertainty import torch_direction

@torch.no_grad()
def build(cohort):
    start=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];d=np.load(ROOT/'runtime/v11/data/train.npz');idx=np.flatnonzero(d['timestamp']<cut);nu=len(s['user.weight']);h=torch.diag(torch.tensor([.25]*8+[1.]*9)).repeat(nu,1,1);im=torch_direction(s['item.weight'],s['item_logstd.weight'].clamp(-3,1).exp());mu=s['user.weight'];norm=mu.norm(dim=1).clamp_min(1e-5);uv=mu/norm[:,None];beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()/math.sqrt(8)
    for st in range(0,len(idx),2048):
        ix=idx[st:st+2048];c=torch.tensor(d['items'][ix],dtype=torch.long);u=torch.tensor(d['user'][ix],dtype=torch.long);dot=(im[c]*uv[u,None]).sum(-1);z=beta*dot+eb*(s['exposure_items'][c]*s['exposure_user.weight'][u,None]).sum(-1)+s['item_bias.weight'][c,0]+s['outside.weight'][u];p=torch.softmax(torch.cat([z,torch.zeros(len(u),1)],1),1)[:,:10];pref=beta*(im[c]-dot[:,:,None]*uv[u,None])/norm[u,None,None];expo=eb*s['exposure_items'][c];grad=torch.cat([pref,expo,torch.ones(len(u),10,1)],2);mean=torch.einsum('nk,nkd->nd',p,grad);fisher=torch.einsum('nk,nkd,nke->nde',p,grad,grad)-mean[:,:,None]*mean[:,None,:];h.index_add_(0,u,fisher)
    corr=torch.empty_like(h);sd=torch.empty(nu);minimum=1.
    for st in range(0,nu,1024):
        cov=torch.linalg.inv(h[st:st+1024]);scale=cov.diagonal(dim1=1,dim2=2).sqrt();r=cov/(scale[:,:,None]*scale[:,None,:]);r=.999*((r+r.transpose(1,2))/2)+.001*torch.eye(17)[None];corr[st:st+1024]=r;sd[st:st+1024]=scale[:,-1];minimum=min(minimum,float(torch.linalg.eigvalsh(r).min()))
    assert minimum>0;np.savez_compressed(folder/'covariance.npz',correlation=corr.numpy(),outside_std=sd.numpy());m.write(folder/'fit.json',dict(cohort=cohort,cutoff=cut,events=len(idx),maximum_training_timestamp=int(d['timestamp'][idx].max()),outside_std_quantiles=np.quantile(sd.numpy(),[0,.5,.9,1]).tolist(),minimum_correlation_eigenvalue=minimum,query_outcomes_used=False,warm_curvature_used=False,seconds=time.time()-start,source_sha256=m.sha(Path(__file__)),plan_sha256=m.sha(ROOT/'experiments/v147/plan.json')))

@torch.no_grad()
def predict(cohort):
    start=time.time();folder=OUT/f'cohort{cohort}';s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];cov=np.load(folder/'covariance.npz');r=torch.tensor(cov['correlation']);control=r.clone();control[:,:16,16]=0;control[:,16,:16]=0;std=torch.cat([s['user_logstd.weight'].clamp(-3,1).exp().expand(-1,8),s['exposure_logstd.weight'].clamp(-4,1).exp().expand(-1,8),torch.tensor(cov['outside_std'])[:,None]],1);chol={str(v):torch.linalg.cholesky(v*r+(1-v)*control)*std[:,:,None] for v in ([0.,.5,1.] if cohort==0 else [0.,1.])};d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(d['user']))
    else:
        idx=np.load(ROOT/'runtime/v142/secondary_evidence.npz')['indices'];idx=idx[(d['timestamp'][idx]>=cut)&(d['timestamp'][idx]<cut+86400)]
    engine=torch.quasirandom.SobolEngine(97,scramble=True,seed=147026);noise=torch.erfinv(2*engine.draw(128).clamp(1e-6,1-1e-6)-1)*math.sqrt(2);eps=torch.cat([noise,-noise]);beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()/math.sqrt(8);parts={k:[] for k in chol}
    for st in range(0,len(idx),128):
        ix=idx[st:st+128];c=torch.tensor(d['items'][ix],dtype=torch.long);u=torch.tensor(d['user'][ix],dtype=torch.long);iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*eps[None,:,17:].reshape(1,len(eps),10,8),dim=-1)
        for name,L in chol.items():
            e=torch.einsum('nde,se->nsd',L[u],eps[:,:17]);uv=torch.nn.functional.normalize(s['user.weight'][u,None]+e[:,:,:8],dim=-1);ex=s['exposure_user.weight'][u,None]+e[:,:,8:16];z=beta*(iv*uv[:,:,None]).sum(-1)+eb*torch.einsum('nkd,nsd->nsk',s['exposure_items'][c],ex)+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]+e[:,:,16,None];p=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2).mean(1).numpy();parts[name].append(p)
    np.savez_compressed(folder/'predictions.npz',indices=idx,**{k:np.concatenate(v).astype(float) for k,v in parts.items()});print(dict(stage='predict',cohort=cohort,queries=len(idx),seconds=time.time()-start),flush=True)

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for correlation in [.5,1.]:
        for mode in ['choice','joint']:
            for strength in [.5,1.]:
                p=m.transfer(base,v['0.0'],v[str(correlation)],mode,strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(correlation=correlation,mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{correlation}_{mode}_{strength}']=p;print(row,flush=True)
    m.write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':build(a.cohort);predict(a.cohort)
    else:screen()
