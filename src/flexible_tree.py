"""Regularised event or binary residual trees with explicit initial offsets."""
import numpy as np,lightgbm as lgb
from scipy.special import softmax,expit


def probability(z,mode):
    z=np.asarray(z).reshape(-1,10)
    if mode=='event':return softmax(np.c_[z,np.zeros(len(z))],axis=1)[:,:10].ravel()
    return expit(z).ravel()


def pairwise_terms(z,positive,negative_pairs,weight):
    """Derivatives of weight*N*mean(softplus(z_negative-z_positive))."""
    slope=expit(z[negative_pairs]-z[positive,None]);curvature=slope*(1-slope)
    scale=weight*len(z)/negative_pairs.size
    gradient=np.bincount(negative_pairs.ravel(),weights=slope.ravel(),minlength=len(z)).astype(np.float64)
    hessian=np.bincount(negative_pairs.ravel(),weights=curvature.ravel(),minlength=len(z)).astype(np.float64)
    gradient[positive]-=slope.sum(1);hessian[positive]+=curvature.sum(1)
    return scale*gradient,scale*hessian


def fit(raw,labels,names,config):
    x=np.asarray(raw,np.float32)
    if np.isinf(x).any():
        x=x.copy();x[np.isinf(x)]=np.nan
    y=np.asarray(labels,np.float64)
    column=names.index(config.get('offset_name','posterior1.0_mix1.0_'+('utility' if config['mode']=='event' else 'logit')))
    offset=x[:,column].astype(np.float64)
    if len(y)%10 or not np.isfinite(offset).all() or (y.reshape(-1,10).sum(1)>1).any():raise ValueError('Invalid event data.')
    first=True;rng=np.random.default_rng(config['seed']);positive=np.flatnonzero(y);negative=np.flatnonzero(y==0)
    def objective(pred,data):
        nonlocal first
        if first:
            if not np.allclose(pred,offset,atol=1e-10,rtol=1e-10):raise ValueError('Missing initial offset.')
            first=False
        p=probability(pred,config['mode']);gradient=p-y;hessian=p*(1-p)
        if config['mode']=='rank_binary' and len(positive) and len(negative):
            pairs=negative[rng.integers(len(negative),size=(len(positive),16))]
            g,h=pairwise_terms(pred,positive,pairs,.25);gradient+=g;hessian+=h
        return gradient,np.maximum(hessian,1e-4)
    params={'objective':objective,'metric':'None','learning_rate':config.get('lr',.025),'num_leaves':config['leaves'],'min_data_in_leaf':config.get('min_leaf',1000),
            'lambda_l2':config.get('l2',100.),'max_bin':127,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,
            'num_threads':config.get('threads',2),'seed':config['seed'],'verbosity':-1,'force_col_wise':True,'deterministic':True}
    data=lgb.Dataset(x,label=y,init_score=offset,free_raw_data=True);model=lgb.train(params,data,num_boost_round=config['rounds'])
    return model,{'config':config,'offset_column':column,'names':names,'binary_event_mass_projection':config['mode']!='event',
                  'pairwise_rank_weight':.25 if config['mode']=='rank_binary' else 0.,'sampled_negatives_per_positive':16 if config['mode']=='rank_binary' else 0}


def predict(model,raw,spec):
    x=np.asarray(raw,np.float32)
    if np.isinf(x).any():
        x=x.copy();x[np.isinf(x)]=np.nan
    z=model.predict(x,raw_score=True,num_threads=spec['config'].get('threads',2))+x[:,spec['offset_column']]
    pred=probability(z,spec['config']['mode']).reshape(-1,10)
    if spec['binary_event_mass_projection']:pred/=np.maximum(pred.sum(1,keepdims=True),1.)
    return pred.ravel()
