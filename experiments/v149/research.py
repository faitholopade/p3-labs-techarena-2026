"""Deterministic full-batch item-context optimization with analytic gradients."""
import argparse,importlib.util,json,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v149'
spec=importlib.util.spec_from_file_location('v148_support',ROOT/'experiments/v148/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);support=m.support

def problem(items,context,source,target,offset,ni,k,precision):
    N=len(items);row=np.arange(N);flat=items.ravel();warm=np.flatnonzero(source);warmflat=items[warm].ravel();code=(items*k+context[:,None]%k).ravel() if k else None
    def objective(w):
        b=w[:ni];bw=w[ni:ni*2];theta=w[ni*2:].reshape(ni,k) if k else None;z=offset+b[items];z[warm]+=bw[items[warm]]
        if k:z=z+(theta-theta.mean(1,keepdims=True)).ravel()[code].reshape(-1,10)
        maxval=z.max(1);p=np.exp(z-maxval[:,None]);den=p.sum(1);loss=(np.log(den)+maxval-z[row,target]).sum()+10*(b@b+bw@bw)+(precision*.5*np.sum(theta*theta) if k else 0);p/=den[:,None];p[row,target]-=1;gb=np.bincount(flat,weights=p.ravel(),minlength=ni)+20*b;gw=np.bincount(warmflat,weights=p[warm].ravel(),minlength=ni)+20*bw
        if k:
            gt=np.bincount(code,weights=p.ravel(),minlength=ni*k).reshape(ni,k);gt-=gt.mean(1,keepdims=True);gt+=precision*theta;g=np.r_[gb,gw,gt.ravel()]
        else:g=np.r_[gb,gw]
        return float(loss/N),g/N
    return objective

def gradient_check():
    rng=np.random.default_rng(149026);c=rng.integers(0,20,(40,10));ctx=rng.integers(0,8,40);source=np.arange(40)>=20;y=rng.integers(0,10,40);offset=rng.normal(size=(40,10));f=problem(c,ctx,source,y,offset,20,4,8);w=rng.normal(size=120)*.03;_,g=f(w);errors=[]
    for j in [0,9,20,39,40,62,90,119]:
        a=w.copy();b=w.copy();a[j]+=1e-5;b[j]-=1e-5;numeric=(f(a)[0]-f(b)[0])/2e-5;errors.append(abs(numeric-g[j]))
    assert max(errors)<1e-8;support.write(OUT/'gradient_check.json',dict(maximum_absolute_error=max(errors),coordinates_checked=len(errors)))

def fit(cohort):
    began=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];factor=m.m.arrays(s);d=np.load(ROOT/'runtime/v11/data/train.npz');past=d['timestamp']<cut;f={k:v[past] for k,v in d.items()};mask=(f['hist']>0)&(f['hist_ts']<1737331200);wu=np.broadcast_to(f['user'][:,None],mask.shape)[mask];wi=f['hist'][mask];wt=f['hist_ts'][mask];_,ix=np.unique(np.rec.fromarrays([wu,wi,wt],names='u,i,t'),return_index=True);wu=wu[ix];wi=wi[ix];wt=wt[ix];wc=m.m.context(wt);real=f['target']<10;ri=f['items'][real];nr=len(ri);pool=dict(np.load(ROOT/m.POOLS[cohort]));rng=np.random.default_rng(149026);items=np.r_[ri,np.c_[wi,support.sample_pool(pool,wu,rng,9)]].astype(np.int32);users=np.r_[f['user'][real],wu];ctx=np.r_[f['context'][real],wc];target=np.r_[f['target'][real],np.zeros(len(wu),int)];source=np.arange(len(items))>=nr;offset=np.empty(items.shape,np.float32)
    for st in range(0,len(items),4096):
        c=items[st:st+4096];u=users[st:st+4096];offset[st:st+4096]=np.einsum('nkd,nd->nk',factor['catalog'][c],factor['user'][u])+factor['bias'][c]
    ni=len(factor['catalog']);configs=[('control',0,0),('k4_p8',4,8),('k4_p32',4,32)]
    if cohort>0:configs=[x for x in configs if x[0] in ['control',json.loads((OUT/'selection.json').read_text())['model']]]
    records=[]
    for name,k,precision in configs:
        start=time.time();objective=problem(items,ctx,source,target,offset,ni,k,precision);result=minimize(objective,np.zeros(ni*(2+k)),jac=True,method='L-BFGS-B',options=dict(maxiter=40,maxcor=6,gtol=1e-8,ftol=1e-11,maxls=20));assert np.isfinite(result.fun) and np.isfinite(result.x).all();w=result.x;model=dict(b=w[:ni],warm=w[ni:ni*2],theta=w[ni*2:].reshape(ni,k),contexts=k,precision=precision);np.savez_compressed(folder/(name+'.npz'),**model);row=dict(model=name,loss=float(result.fun),iterations=int(result.nit),evaluations=int(result.nfev),converged=bool(result.success),message=str(result.message),maximum_gradient=float(abs(result.jac).max()),seconds=time.time()-start);records.append(row);print(row,flush=True)
    support.write(folder/'fit.json',dict(cohort=cohort,cutoff=cut,clicked_real_events=nr,warm_clicks=len(wu),maximum_training_timestamp=int(f['timestamp'].max()),maximum_warm_timestamp=int(wt.max()),records=records,query_outcomes_used=False,source_sha256=support.sha(Path(__file__)),plan_sha256=support.sha(ROOT/'experiments/v149/plan.json'),seconds=time.time()-began))

def predict(cohort):
    folder=OUT/f'cohort{cohort}';d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(d['user']))
    else:
        idx=np.load(ROOT/'runtime/v142/secondary_evidence.npz')['indices'];idx=idx[(d['timestamp'][idx]>=cut)&(d['timestamp'][idx]<cut+86400)]
    c=d['items'][idx];ctx=d['context'][idx];control=np.load(folder/'control.npz')['b'];values={}
    for row in json.loads((folder/'fit.json').read_text())['records']:
        name=row['model']
        if name=='control':continue
        model=np.load(folder/(name+'.npz'));theta=model['theta'];theta=theta-theta.mean(1,keepdims=True);values[name]=model['b'][c]-control[c]+theta[c,ctx[:,None]%int(model['contexts'])]
    np.savez_compressed(folder/'predictions.npz',indices=idx,**values)

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for name in v.files:
        if name=='indices':continue
        for strength in [.25,.5,1.]:
            p=m.transfer(base,v[name],strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(model=name,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{name}_{strength}']=p;print(row,flush=True)
    support.write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':gradient_check();fit(a.cohort);predict(a.cohort)
    else:screen()
