"""Boosted utility corrections with one outside option per ten-candidate event."""
import numpy as np
import lightgbm as lgb
from scipy.special import softmax


def probabilities(utility):
    z=np.asarray(utility).reshape(-1,10)
    return softmax(np.c_[z,np.zeros(len(z))],axis=1)[:,:10].ravel()


def fit(raw,labels,offset_column,rounds=500,leaves=15,seed=20261102,threads=2,external_offset=None):
    x=np.asarray(raw,np.float32);x=np.where(np.isfinite(x),x,np.nan)
    y=np.asarray(labels,np.float64);offset=x[:,offset_column].astype(np.float64) if external_offset is None else np.asarray(external_offset,np.float64)
    if external_offset is not None:x=np.c_[x,offset.astype(np.float32)]
    if len(y)%10 or not np.isfinite(offset).all() or (y.reshape(-1,10).sum(1)>1).any():raise ValueError('Invalid event data.')
    first=True
    def objective(pred,data):
        nonlocal first
        if first:
            if not np.allclose(pred,offset,atol=1e-10,rtol=1e-10):raise ValueError('Initial utility offset not applied by LightGBM.')
            first=False
        probability=probabilities(pred)
        return probability-y,np.maximum(probability*(1-probability),1e-4)
    params={'objective':objective,'metric':'None','learning_rate':.025,'num_leaves':leaves,'min_data_in_leaf':1000,
            'lambda_l2':100.,'max_bin':127,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,
            'num_threads':threads,'seed':seed,'verbosity':-1,'force_col_wise':True,'deterministic':True}
    dataset=lgb.Dataset(x,label=y,init_score=offset,free_raw_data=True)
    model=lgb.train(params,dataset,num_boost_round=rounds)
    return model,{**{k:v for k,v in params.items() if k!='objective'},'objective':'event_softmax_diagonal_hessian','rounds':rounds,'offset_column':offset_column,'prediction_adds_offset':True,'external_offset':external_offset is not None}


def predict(model,raw,spec,external_offset=None):
    x=np.asarray(raw,np.float32);x=np.where(np.isfinite(x),x,np.nan)
    if spec.get('external_offset',False)!=(external_offset is not None):raise ValueError('External offset requirement mismatch.')
    offset=x[:,spec['offset_column']] if external_offset is None else np.asarray(external_offset,np.float64)
    if external_offset is not None:x=np.c_[x,offset.astype(np.float32)]
    utility=model.predict(x,raw_score=True,num_threads=spec.get('num_threads',2))+offset
    return probabilities(utility)
