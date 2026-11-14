"""Crossfit exposure-user prior enhancement using candidate-display likelihood."""
import argparse,importlib.util,json,math,time
from pathlib import Path
import numpy as np,torch,scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v144'
spec=importlib.util.spec_from_file_location('v143_support',ROOT/'experiments/v143/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);support=m.support

def fit(cohort):
    folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];d=np.load(ROOT/'runtime/v11/data/train.npz');past=d['timestamp']<cut;u=d['user'][past];c=d['items'][past];nu=len(s['user.weight']);q=s['exposure_items'].numpy();count=np.bincount(u,minlength=nu);imp=sp.coo_matrix((np.ones(c.size,np.float32),(np.repeat(u,10),c.ravel())),shape=(nu,len(q))).tocsr();profile=imp@q/np.maximum(10*count[:,None],1);v=torch.load(ROOT/f'runtime/v143/cohort{cohort}/anchor80.pt',weights_only=False,map_location='cpu')['display_user'].numpy();norm=np.linalg.norm(v,axis=1,keepdims=True);x=np.c_[profile,v,v/(1+norm),v*norm];y=s['exposure_user.weight'].numpy();priors=np.zeros((2,nu,8));fold=np.arange(nu)%5;coefs={}
    for f in range(5):
        train=(count>=10)&(fold!=f);test=fold==f
        for j,xx in enumerate([profile,x]):
            model=Ridge(alpha=10).fit(xx[train],y[train]);priors[j,test]=model.predict(xx[test]);coefs[f'coef_{j}_{f}']=model.coef_;coefs[f'intercept_{j}_{f}']=model.intercept_
    change=s['exposure_logstd.weight'].clamp(-4,1).mul(2).exp().numpy()/.5*(priors[1]-priors[0]);np.savez_compressed(folder/'prior.npz',control=priors[0],new=priors[1],mean_change=change,**coefs);s['exposure_user.weight']=s['exposure_user.weight']+torch.tensor(change,dtype=torch.float32);torch.save(dict(state=s,cut=cut),folder/'model.pt');m.write(folder/'fit.json',dict(cutoff=cut,events=int(past.sum()),maximum_training_timestamp=int(d['timestamp'][past].max()),prior_mapping_excludes_target_user=True,mean_change_norm_quantiles=np.quantile(np.linalg.norm(change,axis=1),[0,.5,.9,.99,1]).tolist(),query_outcomes_used=False,source_sha256=m.sha(Path(__file__)),plan_sha256=m.sha(ROOT/'experiments/v144/plan.json')))

@torch.no_grad()
def predict(cohort):
    folder=OUT/f'cohort{cohort}';s=torch.load(folder/'model.pt',weights_only=False,map_location='cpu')['state'];ref=np.load(ROOT/f'runtime/v143/cohort{cohort}/predictions.npz');idx=ref['indices'];d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));eps=support.noise();beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp();parts=[]
    for st in range(0,len(idx),256):
        ix=idx[st:st+256];c=torch.tensor(d['items'][ix],dtype=torch.long);u=torch.tensor(d['user'][ix],dtype=torch.long);iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*eps[None,:,8:88].reshape(1,len(eps),10,8),dim=-1);uv=torch.nn.functional.normalize(s['user.weight'][u,None]+s['user_logstd.weight'][u,None].clamp(-3,1).exp()*eps[None,:,:8],dim=-1);ex=(s['exposure_user.weight'][u,None]+s['exposure_logstd.weight'][u,None].clamp(-4,1).exp()*eps[None,:,88:])/math.sqrt(8);z=(iv*uv[:,:,None]).sum(-1)*beta+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]+eb*torch.einsum('nkd,nsd->nsk',s['exposure_items'][c],ex);p=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2).mean(1).numpy();parts.append(p)
    np.savez_compressed(folder/'predictions.npz',indices=idx,original=ref['original'],alternative=np.concatenate(parts).astype(float))

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for mode in ['choice','joint']:
        for strength in [.5,1.]:
            p=m.transfer(base,v['original'],v['alternative'],mode,strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{mode}_{strength}']=p;print(row,flush=True)
    m.write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
