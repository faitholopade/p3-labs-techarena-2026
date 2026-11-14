"""Portable features for the expanded chronological prediction head."""
from pathlib import Path
import numpy as np,torch,gc
from common import subset
from features import build_state,block
from infer_v3 import load_state
from exposure_features import state as exposure_state,features as exposure_features
from residual_model import load as load_residual,features as residual_features
from posterior_inference import features as posterior_features
from activity import rolling,NAMES as ANAMES
from uncertainty import logstd_summary,numpy_direction

def build(fit,d,cut,ni,nu,model,state_root=None,feature_state=None,exposure_basis=None,posterior_samples=64):
    net=load_residual(model);ss=load_state(feature_state) if feature_state is not None else build_state(fit,cut,ni,nu,state_root);x,names=block(d,np.arange(len(d['user'])),ss);del ss;gc.collect()
    # Context code separates hour from weekend; leave the hour feature in its original units.
    x[:,10]%=4;names[10]='hour_bucket'
    es=exposure_state(fit,ni,nu,cut,exposure_basis if exposure_basis is not None else Path(state_root)/str(cut)/'exposure.npz');xx=[]
    for st in range(0,len(d['user']),4000):a,n=exposure_features(subset(d,np.arange(st,min(st+4000,len(d['user'])))),es);xx.append(a)
    x=np.c_[x,np.concatenate(xx)];names+=n;del es,xx;gc.collect()
    a,n=residual_features(net,d);x=np.c_[x,a];names+=n
    a,n=posterior_features(net.state_dict(),d,samples=posterior_samples);x=np.c_[x,a];names+=n
    state=net.state_dict();c=d['items'];u=d['user'];meta=[]
    def add(name,a):
        if a.ndim==1:a=np.repeat(a[:,None],10,1)
        meta.append(a.ravel());names.append(name)
    for kind,ix in [('user',u),('item',c)]:
        mu=state[kind+'.weight'].numpy()[ix];uncertainties=state[kind+'_logstd.weight'].numpy()[ix];ls=logstd_summary(uncertainties);norm=np.linalg.norm(mu,axis=-1)
        certainty=norm/np.sqrt(norm**2+7*np.exp(2*ls))
        if uncertainties.shape[-1]>1:certainty=np.linalg.norm(numpy_direction(mu,np.exp(np.clip(uncertainties,-3,1))),axis=-1)
        add('new_'+kind+'_norm',norm);add('new_'+kind+'_logstd',ls);add('new_'+kind+'_certainty',certainty)
    add('new_outside',state['outside.weight'].numpy()[u,0]);x=np.c_[x,np.stack(meta,1)]
    users=np.r_[fit['user'],d['user']];ts=np.r_[fit['timestamp'],d['timestamp']];ctx=np.r_[fit['context'],d['context']];target=np.r_[fit['target'],np.full(len(d['user']),-1)];act=rolling(users,ts,ctx,target,np.lexsort((ts,users)),nu)[len(fit['user']):];x=np.c_[x,np.repeat(act,10,0)];names+=['activity_'+n for n in ANAMES]
    assert x.shape[1]==len(names);assert not np.isinf(x).any();return x,names

