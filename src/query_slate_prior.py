"""A past-fitted exposure-prior update using only the current offered slate."""
import math
import numpy as np
from scipy.special import softmax
from sklearn.linear_model import Ridge

def fit_prior(state,fit,alpha=10.,minimum_events=10):
    q=state['exposure_items'].numpy().astype(np.float64);nu=len(state['exposure_user.weight'])
    count=np.bincount(fit['user'],minlength=nu).astype(np.float64);totals=np.zeros((nu,q.shape[1]))
    for start in range(0,len(fit['user']),8192):
        u=fit['user'][start:start+8192];values=q[fit['items'][start:start+8192]].sum(1)
        for j in range(q.shape[1]):totals[:,j]+=np.bincount(u,weights=values[:,j],minlength=nu)
    profile=totals/np.maximum(count[:,None]*10,1)
    dense=np.flatnonzero(count>=minimum_events);target=state['exposure_user.weight'].numpy().astype(np.float64)
    if len(dense)<100:raise ValueError('Insufficient users for the fixed ridge fit')
    model=Ridge(alpha=alpha).fit(profile[dense],target[dense])
    return {'coef':model.coef_,'intercept':model.intercept_,'profile':profile,'count':count,'dense_users':len(dense),'alpha':alpha}

def correction(state,prior,query,shrink=True):
    q=state['exposure_items'].numpy().astype(np.float64);u=query['user'];c=query['items']
    query_profile=q[c].mean(1)
    change=(query_profile-prior['profile'][u])/(prior['count'][u,None]+1)
    delta=change@prior['coef'].T
    if shrink:
        variance=np.exp(2*state['exposure_logstd.weight'].numpy()[u].clip(-4,1))
        delta*=np.clip(variance/.5,0,1)
    beta=float(state['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp())
    return np.einsum('nkd,nd->nk',q[c],delta)*beta/math.sqrt(8)

def adjust(probability,delta,strength):
    p=np.asarray(probability,np.float64).reshape(-1,10)
    if strength==0:return p.ravel().copy()
    outside=1-p.sum(1,keepdims=True)
    if p.min()<=0 or outside.min()<=0:raise ValueError('Expected strictly positive eleven-outcome distribution')
    utility=np.log(p/outside)+strength*delta
    return softmax(np.c_[utility,np.zeros(len(p))],axis=1)[:,:10].ravel()
