"""Continuous intrabucket time preferences with exact centered harmonics."""
import argparse,importlib.util,json,math,time
from pathlib import Path
import numpy as np,torch
from scipy.optimize import minimize
from scipy.special import softmax
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v152'
spec=importlib.util.spec_from_file_location('v141_support',ROOT/'experiments/v141/research.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);support=m.support
def basis(ts):
    t=(ts%86400)/86400;bucket=np.floor(t*4).astype(int);terms=[]
    for frequency in [1,2]:
        omega=2*np.pi*frequency;lo=omega*bucket/4;hi=omega*(bucket+1)/4;terms.extend([np.sin(omega*t)-(np.cos(lo)-np.cos(hi))/(omega/4),np.cos(omega*t)-(np.sin(hi)-np.sin(lo))/(omega/4)])
    return np.stack(terms,1)
def problem(items,ctx,phi,target,offset,catalog,ridge):
    N=len(items);ni=len(catalog);flat=items.ravel();code=(items*4+ctx[:,None]).ravel();row=np.arange(N)
    def objective(w):
        weight=w.reshape(8,16);scores=catalog@weight.T;z=offset+scores[items,ctx[:,None]]
        for j in range(4):z=z+scores[items,j+4]*phi[:,j,None]
        maximum=z.max(1);p=np.exp(z-maximum[:,None]);den=p.sum(1);loss=float((np.log(den)+maximum-z[row,target]).mean())+.5*ridge*float((weight*weight).sum());p/=den[:,None];p[row,target]-=1;gitem=np.bincount(code,weights=p.ravel(),minlength=ni*4).reshape(ni,4);g=np.empty_like(weight);g[:4]=(catalog.T@gitem).T/N
        for j in range(4):g[4+j]=catalog.T@np.bincount(flat,weights=(p*phi[:,j,None]).ravel(),minlength=ni)/N
        g+=ridge*weight;return loss,g.ravel()
    return objective
def gradient_check():
    rng=np.random.default_rng(152026);c=rng.integers(0,25,(40,10));ctx=rng.integers(0,4,40);phi=rng.normal(size=(40,4));y=rng.integers(0,10,40);z=rng.normal(size=(40,10));cat=rng.normal(size=(25,16));f=problem(c,ctx,phi,y,z,cat,.1);w=rng.normal(size=128)*.03;_,g=f(w);error=[]
    for j in [0,17,55,64,82,110,127]:
        a=w.copy();b=w.copy();a[j]+=1e-5;b[j]-=1e-5;error.append(abs((f(a)[0]-f(b)[0])/2e-5-g[j]))
    assert max(error)<1e-8;support.write(OUT/'gradient_check.json',dict(maximum_error=max(error),coordinates=len(error)))
def fit(cohort):
    start=time.time();folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);cut=support.CUTS[cohort];s=torch.load(ROOT/support.FACTORS[cohort],weights_only=False,map_location='cpu')['state'];factor=m.arrays(s);d=np.load(ROOT/'runtime/v11/data/train.npz');idx=np.flatnonzero((d['timestamp']<cut)&(d['target']<10));items=d['items'][idx];users=d['user'][idx];ts=d['timestamp'][idx];ctx=d['context'][idx]%4;phi=basis(ts);target=d['target'][idx];offset=np.empty(items.shape,float)
    for st in range(0,len(idx),4096):
        c=items[st:st+4096];u=users[st:st+4096];offset[st:st+4096]=np.einsum('nkd,nd->nk',factor['catalog'][c],factor['user'][u])+factor['bias'][c]
    raw=factor['catalog'].astype(float);mean=raw[items].mean((0,1));std=np.maximum(raw[items].std((0,1)),1e-4);catalog=np.clip((raw-mean)/std,-6,6);ridges=[.01,.1]
    if cohort>0:ridges=[json.loads((OUT/'selection.json').read_text())['ridge']]
    records=[]
    for ridge in ridges:
        began=time.time();objective=problem(items,ctx,phi,target,offset,catalog,ridge);result=minimize(objective,np.zeros(128),jac=True,method='L-BFGS-B',options=dict(maxiter=70,ftol=1e-12,gtol=1e-8,maxls=30));assert result.success,result.message;w=result.x.reshape(8,16);np.savez_compressed(folder/f'ridge{ridge}.npz',catalog=catalog,weight=w[4:],nuisance=w[:4],mean=mean,std=std,ridge=ridge);row=dict(ridge=ridge,loss=float(result.fun),iterations=int(result.nit),converged=bool(result.success),maximum_gradient=float(abs(result.jac).max()),seconds=time.time()-began);records.append(row);print(row,flush=True)
    support.write(folder/'fit.json',dict(cohort=cohort,cutoff=cut,clicked_real_events=len(idx),maximum_training_timestamp=int(ts.max()),records=records,query_outcomes_used=False,source_sha256=support.sha(Path(__file__)),plan_sha256=support.sha(ROOT/'experiments/v152/plan.json'),seconds=time.time()-start))
def predict(cohort):
    folder=OUT/f'cohort{cohort}';d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if cohort==3 else 'runtime/v11/data/train.npz'));cut=support.CUTS[cohort]
    if cohort==0:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    elif cohort==3:idx=np.arange(len(d['user']))
    else:
        idx=np.load(ROOT/'runtime/v142/secondary_evidence.npz')['indices'];idx=idx[(d['timestamp'][idx]>=cut)&(d['timestamp'][idx]<cut+86400)]
    c=d['items'][idx];phi=basis(d['timestamp'][idx]);values={}
    for row in json.loads((folder/'fit.json').read_text())['records']:
        ridge=row['ridge'];model=np.load(folder/f'ridge{ridge}.npz');scores=model['catalog']@model['weight'].T;raw=np.sum(scores[c]*phi[:,None,:],axis=2);raw-=raw.mean(1,keepdims=True);values[str(ridge)]=.2*np.tanh(raw/.2)
    np.savez_compressed(folder/'predictions.npz',indices=idx,**values)
def transfer(base,d,strength):
    b=base.reshape(-1,10);return (softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)).ravel()
def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for ridge in [.01,.1]:
        for strength in [.25,.5,1.]:
            p=transfer(base,v[str(ridge)],strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(ridge=ridge,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{ridge}_{strength}']=p;print(row,flush=True)
    support.write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':gradient_check();fit(a.cohort);predict(a.cohort)
    else:screen()
