import json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,log_loss
from sklearn.utils.extmath import randomized_svd

def load_data(cache):
    cache=Path(cache); audit=json.loads((cache/'audit.json').read_text())
    return dict(np.load(cache/'train.npz')),dict(np.load(cache/'evaluation.npz')),audit

def split_data(d,cutoff):
    fit=np.flatnonzero(d['timestamp']<cutoff)
    after=np.flatnonzero(d['timestamp']>=cutoff)
    after=after[np.argsort(d['timestamp'][after],kind='stable')]
    _,p=np.unique(d['user'][after],return_index=True)
    valid=after[np.sort(p)]
    return fit,valid

def subset(d,ix):return {k:v[ix] for k,v in d.items()}

def binary_labels(d):return (np.arange(10)[None,:]==d['target'][:,None]).astype(np.float32)

def metrics(y,p):
    p=np.clip(p,1e-7,1-1e-7)
    return {'auc':float(roc_auc_score(y.ravel(),p.ravel())),'logloss':float(log_loss(y.ravel(),p.ravel())),'mean_probability':float(p.mean())}

def stats(d,ni,nu):
    y=binary_labels(d);c=d['items'];u=d['user'];ctx=d['context']
    imp=np.bincount(c.ravel(),minlength=ni+1).astype(np.float32)
    hit=np.bincount(c.ravel(),weights=y.ravel(),minlength=ni+1).astype(np.float32)
    ui=np.bincount(u,minlength=nu+1).astype(np.float32)
    uh=np.bincount(u,weights=(d['target']<10),minlength=nu+1).astype(np.float32)
    cc=(c*8+ctx[:,None]).ravel()
    ci=np.bincount(cc,minlength=(ni+1)*8).reshape(ni+1,8).astype(np.float32)
    ch=np.bincount(cc,weights=y.ravel(),minlength=(ni+1)*8).reshape(ni+1,8).astype(np.float32)
    return dict(imp=imp,hit=hit,ui=ui,uh=uh,ci=ci,ch=ch,g=float(y.mean()) if len(y) else 0.05)

def chronological_features(d,ni,nu):
    """Training encodings use strictly earlier UTC days, never the row's label."""
    result=None
    day=d['timestamp']//86400
    for block in np.unique(day):
        ix=np.flatnonzero(day==block);past=np.flatnonzero(day<block)
        ss=stats(subset(d,past),ni,nu)
        ff=feature_arrays(subset(d,ix),ss,False)
        if result is None:result=[np.empty((len(day),)+f.shape[1:],dtype=f.dtype) for f in ff]
        for dest,source in zip(result,ff):dest[ix]=source
    return result

def feature_arrays(d,s,leave_one_out=False):
    c=d['items'];u=d['user'];ctx=d['context'];n=len(u);y=binary_labels(d) if leave_one_out else 0.0
    sub=1.0 if leave_one_out else 0.0
    rate=(s['hit'][c]-y+50*s['g'])/(s['imp'][c]-sub+50)
    urate=(s['uh'][u]-(d['target']<10)*sub+5*s['g']*10)/(s['ui'][u]-sub+5)
    crate=(s['ch'][c,ctx[:,None]]-y+50*rate)/(s['ci'][c,ctx[:,None]]-sub+50)
    h=d['hist'];mask=h>0;length=mask.sum(1)
    age=np.maximum(0,(d['timestamp'][:,None]-d['hist_ts'])/86400.).astype(np.float32)
    logage=np.log1p(age)*mask
    repeat=(c[:,:,None]==h[:,None,:]) & mask[:,None,:]
    rep=repeat.sum(2).astype(np.float32)
    last=(c==h[:,-1,None]).astype(np.float32)
    since=np.where(repeat,age[:,None,:],1e5).min(2)
    rate=np.clip(rate,1e-5,1-1e-5);crate=np.clip(crate,1e-5,1-1e-5)
    row=np.stack([logit(rate),logit(crate)-logit(rate),np.log1p(rep),last,np.where(rep>0,np.log1p(since),0),np.log1p(s['imp'][c])/10],2).astype(np.float32)
    event=np.stack([logit(np.clip(urate,1e-4,1-1e-4)),length/20,np.log1p(np.maximum(d['timestamp']-d['hist_ts'][:,-1],0)/3600)* (length>0),np.log1p(s['ui'][u])/5],1).astype(np.float32)
    return row,event,age

def interaction_matrix(d,ni,nu):
    h=d['hist'];mask=h>0
    rr=np.broadcast_to(d['user'][:,None],h.shape)[mask]
    cc=h[mask]
    clicked=np.flatnonzero(d['target']<10)
    rr=np.concatenate([rr,d['user'][clicked]])
    cc=np.concatenate([cc,d['items'][clicked,d['target'][clicked]]])
    m=sp.coo_matrix((np.ones(len(rr),np.float32),(rr,cc)),shape=(nu+1,ni+1)).tocsr()
    m.data[:]=1
    return m

def init_svd(m,dim=64):
    ip=np.asarray(m.sum(0)).ravel();up=np.asarray(m.sum(1)).ravel()
    w=sp.diags(1/np.sqrt(np.maximum(up,1)))@m@sp.diags(1/np.sqrt(np.maximum(ip,1)))
    _,ss,vt=randomized_svd(w,n_components=dim+1,n_iter=5,random_state=2718)
    v=(vt[1:].T*np.sqrt(ss[1:])).astype(np.float32)
    v/=np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-6)
    v[0]=0
    return v

def coclick(m):
    pop=np.asarray(m.sum(0)).ravel();co=(m.T@m).tocsr();co.setdiag(0);co.eliminate_zeros()
    scale=1/np.sqrt(np.maximum(pop,1));co=sp.diags(scale)@co@sp.diags(scale)
    return co.tocsr()

def starter_predict(d,s,co):
    c=d['items'];u=d['user'];g=s['g']
    ip=(s['hit'][c]+50*g)/(s['imp'][c]+50)
    up=(s['uh'][u]+50*g)/(10*s['ui'][u]+50)
    z=logit(ip)+logit(up[:,None])-logit(g)
    for st in range(0,len(c),1024):
        en=st+1024;cc=c[st:en];hh=d['hist'][st:en]
        a=np.broadcast_to(cc[:,:,None],(len(cc),10,20)).reshape(-1)
        b=np.broadcast_to(hh[:,None,:],(len(cc),10,20)).reshape(-1)
        cf=np.asarray(co[a,b]).reshape(len(cc),10,20).sum(2)
        z[st:en]+=25*cf
    return expit(z)
