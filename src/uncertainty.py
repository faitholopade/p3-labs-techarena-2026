"""Moment approximations for scalar or diagonal latent uncertainty."""
import numpy as np


def torch_direction(mu,std,exact=False):
    if exact:
        import torch
        if torch.is_grad_enabled() and (mu.requires_grad or std.requires_grad):raise ValueError('Exact direction moments are inference-only.')
        return torch.from_numpy(numpy_direction(mu.detach().cpu().numpy(),std.detach().cpu().numpy(),exact=True)).to(mu)
    norm2=mu.square().sum(-1,keepdim=True)
    if std.shape[-1]==1:
        variance=(mu.shape[-1]-1)*std.square()
    else:
        direction=mu/norm2.sqrt().clamp_min(1e-8)
        variance=(std.square().sum(-1,keepdim=True)-(direction.square()*std.square()).sum(-1,keepdim=True)).clamp_min(0)
    return mu/(norm2+variance).sqrt().clamp_min(1e-8)


def numpy_direction(mu,std,exact=False):
    if exact:
        # Isotropic projected-normal mean: Herrera-Esposito & Burge (2025),
        # section 2.2.3, https://pmc.ncbi.nlm.nih.gov/articles/PMC12447714/.
        from scipy.special import hyp1f1,gammaln
        if std.shape[-1]!=1:raise ValueError('Exact moments require scalar isotropic uncertainty.')
        m=np.asarray(mu,np.float64);s=np.asarray(std,np.float64)
        if (s<=0).any() or not np.isfinite(s).all():raise ValueError('Standard deviations must be positive and finite.')
        d=m.shape[-1];ratio2=np.sum(m*m,axis=-1,keepdims=True)/(s*s)
        coefficient=np.exp(gammaln((d+1)/2)-gammaln(d/2+1))/(np.sqrt(2)*s)
        result=m*coefficient*hyp1f1(.5,d/2+1,-.5*ratio2)
        if not np.isfinite(result).all():raise ValueError('Nonfinite exact direction moment.')
        return result.astype(np.asarray(mu).dtype)
    norm2=(mu*mu).sum(-1,keepdims=True)
    if std.shape[-1]==1:
        variance=(mu.shape[-1]-1)*std*std
    else:
        direction=mu/np.maximum(np.sqrt(norm2),1e-8)
        variance=np.maximum((std*std).sum(-1,keepdims=True)-(direction*direction*std*std).sum(-1,keepdims=True),0)
    return mu/np.maximum(np.sqrt(norm2+variance),1e-8)


def logstd_summary(values,minimum=-3,maximum=1):
    if values.shape[-1]==1:return values[...,0]
    return .5*np.log(np.exp(2*np.clip(values,minimum,maximum)).mean(-1))
