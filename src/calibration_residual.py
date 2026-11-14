"""Outcome-free features and a probability-preserving calibration mixture."""
import numpy as np
from scipy.special import logit


CALIBRATION_WEIGHT=15/64


def mixture(tree,neural):
    return (7*np.asarray(tree)+8*np.asarray(neural))/15


def utility(prob):
    p=np.asarray(prob,dtype=np.float64).reshape(-1,10)
    return (np.log(np.maximum(p,1e-12))-np.log(np.maximum(1-p.sum(1,keepdims=True),1e-12))).ravel()


def features(raw,names,tree,neural):
    selected=list(range(70))+list(range(140,len(names)))
    bad={'item_context_delta','item_context_impressions_log','activity_log_observation_days','activity_last_day_count','activity_last_context_count'}
    selected=[j for j in selected if names[j] not in bad]
    q=mixture(tree,neural);qq=q.reshape(-1,10);event=qq.sum(1);share=qq/np.maximum(event[:,None],1e-12)
    uq=utility(q);difference=utility(tree)-utility(neural)
    added=np.column_stack([uq,logit(np.clip(q,1e-9,1-1e-9)),np.repeat(logit(np.clip(event,1e-9,1-1e-9)),10),
                          np.log(np.maximum(share,1e-12)).ravel(),difference,np.abs(difference),
                          uq-raw[:,names.index('posterior1.0_mix1.0_utility')]])
    labels=[names[j] for j in selected]+['calibration_mix_utility','calibration_mix_logit','calibration_mix_event_logit',
            'calibration_mix_log_share','calibration_head_utility_difference','calibration_head_utility_disagreement','calibration_mix_posterior_delta']
    x=np.empty((len(raw),len(labels)),dtype=np.float32)
    x[:,:len(selected)]=raw[:,selected];x[:,len(selected):]=added
    return x,labels


def correct(base,tree,neural,new,weight):
    if not 0<=weight<=CALIBRATION_WEIGHT:raise ValueError('Invalid replacement weight.')
    return np.asarray(base)+weight*(np.asarray(new)-mixture(tree,neural))
