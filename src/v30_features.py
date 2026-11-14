"""Known query context, appended on demand without copying multi-GB caches."""
import numpy as np

CONTEXT_NAMES=['query_is_weekend']+[f'query_context_{i}' for i in range(8)]

def context_features(context):
    c=np.asarray(context)
    if c.ndim!=1 or not np.isin(c,np.arange(8)).all():raise ValueError('Expected context codes 0 through 7.')
    return np.c_[(c//4).astype(np.float32),np.eye(8,dtype=np.float32)[c.astype(int)]]

class FeatureView:
    def __init__(self,raw,context=None):
        self.raw=raw;self.context=None if context is None else context_features(context)
        if self.context is not None and len(raw)!=10*len(self.context):raise ValueError('Context alignment mismatch.')
        self.shape=(len(raw),344+(0 if self.context is None else 9))
    def __len__(self):return self.shape[0]
    def __getitem__(self,key):
        columns=None
        if isinstance(key,tuple):key,columns=key
        base=self.raw[key,:344]
        if self.context is not None:
            rows=np.arange(*key.indices(len(self))) if isinstance(key,slice) else np.asarray(key)
            base=np.concatenate([base,self.context[rows//10]],axis=-1)
        return base if columns is None else base[:,columns]

def stable_columns(names):
    bad={'item_context_delta','item_context_impressions_log','activity_log_observation_days','activity_last_day_count','activity_last_context_count'}
    return [i for i in list(range(70))+list(range(140,len(names))) if names[i] not in bad]
