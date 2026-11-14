"""Ridge event-rate residual with frozen utilities and invariant slate summaries."""
import numpy as np
from scipy.special import softmax,logsumexp,expit
from scipy.optimize import minimize

NAMES=['utility_std','utility_range','top_utility_gap','conditional_entropy','top_conditional_mass','top3_conditional_mass','conditional_squared_mass','frozen_event_logit']

def features(utility,event_delta):
    u=np.asarray(utility,np.float64);q=softmax(u,axis=1);sorted_u=np.sort(u,axis=1);sorted_q=np.sort(q,axis=1)
    offset=logsumexp(u,axis=1)+np.asarray(event_delta,np.float64)
    x=np.column_stack([u.std(1),np.ptp(u,axis=1),sorted_u[:,-1]-sorted_u[:,-2],-(q*np.log(np.maximum(q,1e-15))).sum(1),sorted_q[:,-1],sorted_q[:,-3:].sum(1),(q*q).sum(1),offset])
    assert np.isfinite(x).all();return x,offset

def fit(x,offset,y,penalty=.02):
    mean=x.mean(0);std=np.maximum(x.std(0),1e-3);z=np.column_stack([np.ones(len(x)),np.clip((x-mean)/std,-8,8)]);y=np.asarray(y,np.float64)
    def objective(w):
        score=offset+z@w;loss=np.mean(np.logaddexp(0,score)-y*score)+.5*penalty*(w[1:]@w[1:]);grad=z.T@(expit(score)-y)/len(y);grad[1:]+=penalty*w[1:];return loss,grad
    result=minimize(objective,np.zeros(z.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-12,'gtol':1e-8})
    assert result.success,result.message
    return dict(mean=mean.tolist(),std=std.tolist(),coef=result.x.tolist(),penalty=penalty,cap=.20,names=NAMES,events=len(y),iterations=int(result.nit),loss=float(result.fun))

def delta(model,x):
    z=np.column_stack([np.ones(len(x)),np.clip((x-np.array(model['mean']))/np.array(model['std']),-8,8)])
    return np.clip(z@np.array(model['coef']),-model['cap'],model['cap'])
