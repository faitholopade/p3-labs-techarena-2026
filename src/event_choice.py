"""Ten mutually exclusive click outcomes plus one no-click outcome."""
import numpy as np
from scipy.special import softmax

OFFSET=-1.77

def probabilities(raw):
    logits=np.asarray(raw).reshape(-1,10)+OFFSET
    return softmax(np.c_[logits,np.zeros(len(logits))],axis=1)[:,:10].ravel()

def objective(raw,data):
    p=probabilities(raw)
    return p-data.get_label(),np.maximum(p*(1-p),1e-5)

def combine(v5,event_choice,weight=.5):
    base=np.asarray(v5).reshape(-1,10)
    base=base/np.maximum(base.sum(1,keepdims=True),1)
    return ((1-weight)*base+weight*np.asarray(event_choice).reshape(-1,10)).ravel()
