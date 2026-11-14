"""Event click propensity from strictly prior activity and history-slate summaries."""
import numpy as np
from scipy.special import logit,expit,logsumexp,softmax
from history_choice import features as history_features
from activity import rolling,NAMES


def build(fit,query,items,nu,training_path=None):
    n=len(fit['user']);data={k:np.concatenate([fit[k],query[k]]) for k in fit}
    if not np.all(query['target']==-1):raise ValueError('Query outcomes must be masked.')
    if fit['timestamp'].max()>=query['timestamp'].min():raise ValueError('Chronological cutoff violation.')
    a=rolling(data['user'],data['timestamp'],data['context'],data['target'],np.lexsort((data['timestamp'],data['user'])),nu)
    # Absolute observation-window position was excluded before viewing results.
    selected=[j for j in range(32) if j!=28];out=None;names=None
    for st in range(0,len(data['user']),2048):
        sl=slice(st,st+2048);x,base,fnames=history_features({k:v[sl] for k,v in data.items()},items)
        conditional=softmax(base,1);entropy=-(conditional*np.log(np.maximum(conditional,1e-12))).sum(1)
        block=np.c_[a[sl][:,selected],x.mean(1),x.std(1),x.max(1),logsumexp(base,1),entropy]
        labels=['activity_'+NAMES[j] for j in selected]+[name+'_'+summary for summary in ['mean','std','max'] for name in fnames]+['history_log_partition','history_choice_entropy']
        if out is None:
            names=labels;shape=(len(data['user']),len(names))
            out=np.lib.format.open_memmap(training_path,mode='w+',shape=shape,dtype=np.float32) if training_path is not None else np.empty(shape,np.float32)
        if labels!=names or not np.isfinite(block).all():raise ValueError('Feature contract failed.')
        out[st:st+len(block)]=block
    if hasattr(out,'flush'):out.flush()
    return out,logit(np.clip(a[:,2],1e-6,1-1e-6)),names


def predict(model,x,offset,threads=2):return expit(model.predict(x,raw_score=True,num_threads=threads)+offset)
