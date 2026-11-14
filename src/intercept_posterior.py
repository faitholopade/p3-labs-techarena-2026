"""One-dimensional posterior for a user's click intercept, given past slates."""
import numpy as np,torch
from scipy.special import expit,logsumexp
from numpy.polynomial.hermite import hermgauss


@torch.no_grad()
def log_mass(net,events,batch=8192):
    out=np.empty(len(events['user']),np.float64)
    for start in range(0,len(out),batch):
        end=min(start+batch,len(out));u=torch.tensor(events['user'][start:end]);c=torch.tensor(events['items'][start:end]);ctx=torch.tensor(events['context'][start:end])
        z=net(c,u,False,ctx)-net.outside(u)
        out[start:end]=torch.logsumexp(z,dim=1).numpy()
    return out


def fit(users,mass,clicked,prior_mean,precision=.25,nodes=31):
    users=np.asarray(users);mass=np.asarray(mass,np.float64);y=np.asarray(clicked,np.float64);prior=np.asarray(prior_mean,np.float64);nu=len(prior)
    mode=prior.copy()
    for _ in range(18):
        probability=expit(mass+mode[users]);gradient=np.bincount(users,weights=y-probability,minlength=nu)-precision*(mode-prior)
        curvature=np.bincount(users,weights=probability*(1-probability),minlength=nu)+precision
        step=np.clip(gradient/curvature,-2,2);mode+=step
        if np.max(abs(step))<1e-8:break
    probability=expit(mass+mode[users]);curvature=np.bincount(users,weights=probability*(1-probability),minlength=nu)+precision
    x,w=hermgauss(nodes);grid=mode[:,None]+np.sqrt(2/curvature[:,None])*x
    logp=-.5*precision*(grid-prior[:,None])**2
    for start in range(0,len(users),4096):
        end=min(start+4096,len(users));u=users[start:end];z=mass[start:end,None]+grid[u]
        ll=y[start:end,None]*z-np.logaddexp(0,z);np.add.at(logp,u,ll)
    logw=logp+x[None,:]**2+np.log(w)[None,:];weights=np.exp(logw-logsumexp(logw,axis=1,keepdims=True))
    mean=(weights*grid).sum(1);variance=(weights*(grid-mean[:,None])**2).sum(1)
    assert np.isfinite(weights).all() and np.max(abs(weights.sum(1)-1))<1e-10
    return {'grid':grid,'weight':weights,'mode':mode,'mean':mean,'variance':variance,'precision':precision,'nodes':nodes}


def predict(posterior,users,mass):
    users=np.asarray(users);p=expit(np.asarray(mass)[:,None]+posterior['grid'][users]);w=posterior['weight'][users]
    mean=(p*w).sum(1);variance=(w*(p-mean[:,None])**2).sum(1)
    return mean,variance
