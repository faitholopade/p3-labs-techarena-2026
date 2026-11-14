"""Weekend preferences learned from pre-cutoff real and warm-up clicks."""
import argparse,gc,json,math,sys,time,importlib.util
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v141'
spec=importlib.util.spec_from_file_location('v138_support',ROOT/'experiments/v138/research.py');support=importlib.util.module_from_spec(spec);spec.loader.exec_module(support)
sys.path.insert(0,str(ROOT/'src'));from uncertainty import torch_direction
write=support.write;sha=support.sha

def context(t):return (t%86400//21600+4*((t//86400+3)%7>=5)).astype(int)

@torch.no_grad()
def arrays(s):
    iv=torch_direction(s['item.weight'],s['item_logstd.weight'].clamp(-3,1).exp()).numpy();uv=torch_direction(s['user.weight'],s['user_logstd.weight'].clamp(-3,1).exp()).numpy();beta=float(s['log_beta'].clamp(math.log(2),math.log(8)).exp());eb=float(s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp())/math.sqrt(8)
    return dict(catalog=np.c_[iv,s['exposure_items'].numpy()],user=np.c_[uv*beta,s['exposure_user.weight'].numpy()*eb],bias=s['item_bias.weight'].numpy()[:,0])

def fit(cohort):
    start=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];ck=torch.load(ROOT/support.FACTORS[cohort],map_location='cpu',weights_only=False);assert ck['cut']==cut;factors=arrays(ck['state']);data=dict(np.load(ROOT/'runtime/v11/data/train.npz'));idx=np.flatnonzero(data['timestamp']<cut);f={k:v[idx] for k,v in data.items()};del data
    mask=(f['hist']>0)&(f['hist_ts']<1737331200);u=np.broadcast_to(f['user'][:,None],mask.shape)[mask];i=f['hist'][mask];t=f['hist_ts'][mask];_,ix=np.unique(np.rec.fromarrays([u,i,t],names='u,i,t'),return_index=True);wu=u[ix];wi=i[ix];wt=t[ix];pool=dict(np.load(ROOT/support.POOLS[cohort]));rng=np.random.default_rng(141026);clicked=f['target']<10;realitems=f['items'][clicked];warmitems=np.c_[wi,support.sample_pool(pool,wu,rng,9)];items=np.r_[realitems,warmitems];users=np.r_[f['user'][clicked],wu];ctx=np.r_[f['context'][clicked],context(wt)];source=np.r_[np.zeros(len(realitems),int),np.ones(len(wu),int)];target=np.r_[f['target'][clicked],np.zeros(len(wu),int)];assert np.array_equal(context(f['timestamp']),f['context']);catalog=factors['catalog'];mean=catalog[items].mean((0,1));std=np.maximum(catalog[items].std((0,1)),1e-4);x=np.clip((catalog[items]-mean)/std,-6,6).astype(np.float64);x-=x.mean(1,keepdims=True);offset=(np.einsum('nkd,nd->nk',catalog[items],factors['user'][users])+factors['bias'][items]).astype(np.float64);groups=[]
    for so in range(2):
        for c in range(8):
            m=(source==so)&(ctx==c)
            if m.any():groups.append((so,c,np.ascontiguousarray(x[m].reshape(-1,16)),offset[m],target[m]))
    del x,offset,items,users;gc.collect();n=len(target)
    for variant in ['shared','byhour']:
        began=time.time();nw=1 if variant=='shared' else 4;shape=(8+nw,16);ridge=.03;calls=0
        def objective(flat):
            nonlocal calls
            w=flat.reshape(shape);g=np.zeros_like(w);loss=.5*ridge*float((w*w).sum())
            for so,c,x,z0,yy in groups:
                q=(c//4)-2/7;nu=so*4+c%4;week=8+(0 if nw==1 else c%4);coef=w[nu]+q*w[week];h=np.tanh((x@coef).reshape(-1,10)/.3);z=z0+.3*h;prob=softmax(z,axis=1);loss+=float((logsumexp(z,axis=1)-z[np.arange(len(z)),yy]).sum())/n;prob[np.arange(len(z)),yy]-=1;grad=x.T@(prob*(1-h*h)).ravel()/n;g[nu]+=grad;g[week]+=q*grad
            g+=ridge*w;calls+=1;return loss,g.ravel()
        result=minimize(objective,np.zeros(np.prod(shape)),jac=True,method='L-BFGS-B',options=dict(maxiter=70,ftol=1e-12,gtol=1e-7,maxls=30));assert result.success,result.message;weight=result.x.reshape(shape)[8:];np.savez_compressed(folder/f'{variant}.npz',mean=mean,std=std,weight=weight,nuisance=result.x.reshape(shape)[:8],catalog=catalog);report=dict(cohort=cohort,variant=variant,cutoff=cut,clicked_real_events=len(realitems),warm_events=len(wu),warm_weekend_events=int((context(wt)>=4).sum()),maximum_training_timestamp=int(f['timestamp'].max()),maximum_warm_timestamp=int(wt.max()),iterations=int(result.nit),calls=calls,loss=float(result.fun),converged=bool(result.success),query_outcomes_used=False,factor_sha256=sha(ROOT/support.FACTORS[cohort]),source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'experiments/v141/plan.json'),seconds=time.time()-began);write(folder/f'{variant}_fit.json',report);print(report,flush=True)

def delta(model,items,ctx):
    x=np.clip((model['catalog'][items]-model['mean'])/model['std'],-6,6);x-=x.mean(1,keepdims=True);w=model['weight'][np.zeros(len(ctx),int) if len(model['weight'])==1 else ctx%4];linear=np.einsum('nkd,nd->nk',x,w)*((ctx//4)-2/7)[:,None];return .2*np.tanh(linear/.2)

def predict(cohort):
    folder=OUT/f'cohort{cohort}';data=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v136/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(data['user']))
    else:
        idx=np.load(ROOT/'runtime/v136/secondary_evidence.npz')['indices'];idx=idx[(data['timestamp'][idx]>=cut)&(data['timestamp'][idx]<cut+86400)]
    values={v:delta(np.load(folder/f'{v}.npz'),data['items'][idx],data['context'][idx]) for v in ['shared','byhour']};np.savez_compressed(folder/'predictions.npz',indices=idx,**values)

def transfer(base,d,strength):
    b=base.reshape(-1,10);return (softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)).ravel()

def screen():
    e=np.load(ROOT/'runtime/v136/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');assert np.array_equal(e['indices'],v['indices']);base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};rows=[];pred={}
    for variant in ['shared','byhour']:
        for strength in [.25,.5,1.]:
            p=transfer(base,v[variant],strength);g={k:float(roc_auc_score(y[m],p[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(variant=variant,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{variant}_{strength}']=p;print(row,flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
