"""Estimate event click propensity separately from conditional item choice."""
import numpy as np,lightgbm as lgb
from scipy.special import expit,logit


def aggregate(raw,names):
    # Fixed feature selection and outcome-free summaries shared by all folds.
    selected=list(range(35))+list(range(209,len(names)))
    x=np.asarray(raw[:,selected],np.float32).reshape(-1,10,len(selected))
    out=np.c_[np.nanmean(x,1),np.nanstd(x,1),np.nanmax(x,1)]
    labels=[names[j]+'_'+summary for summary in ['mean','std','max'] for j in selected]
    rate=np.asarray(raw[:,names.index('posterior1.0_mix1.0_event')]).reshape(-1,10)[:,0]
    return out,logit(np.clip(rate,1e-7,1-1e-7)),labels


def fit(x,labels,offset,names,config):
    x=np.where(np.isfinite(x),x,np.nan).astype(np.float32);offset=np.asarray(offset,np.float64)
    params={'objective':'binary','metric':'None','learning_rate':.025,'num_leaves':7,'min_data_in_leaf':500,'lambda_l2':100.,
            'max_bin':127,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'num_threads':2,'seed':20261103,
            'verbosity':-1,'force_col_wise':True,'deterministic':True}
    params.update(config)
    model=lgb.train(params,lgb.Dataset(x,label=labels,init_score=offset,free_raw_data=True),num_boost_round=600)
    return model,{'params':params,'rounds':600,'names':names,'initial_offset':'logit(posterior1.0_mix1.0_event)','prediction_adds_offset':True}


def predict(model,x,offset,spec):
    x=np.where(np.isfinite(x),x,np.nan).astype(np.float32)
    return expit(model.predict(x,raw_score=True,num_threads=spec['params']['num_threads'])+offset)
