"""Outcome-free candidate-history and slate summaries from a frozen factor."""
import numpy as np
from uncertainty import numpy_direction


def extend(raw,events,state,names):
    if len(raw)!=10*len(events['user']):raise ValueError('Event and row counts differ.')
    v=numpy_direction(state['item.weight'].numpy(),state['item_logstd.weight'].clamp(-3,1).exp().numpy());v[0]=0
    q=state['exposure_items'].numpy()[:,:8].copy();q[0]=0
    chunks=[];extra_names=None
    for start in range(0,len(events['user']),512):
        sl=slice(start,start+512);c=events['items'][sl];h=events['hist'][sl];mask=h>0
        age=(events['timestamp'][sl,None]-events['hist_ts'][sl])/86400
        if not np.all(age[mask]>0):raise ValueError('History must precede the query.')
        count=mask.sum(1).clip(1);cols=[];labels=[]
        def add(name,value):cols.append(value.reshape(-1));labels.append(name)
        for label,vec in [('newpref',v),('newexposure',q)]:
            score=np.einsum('nkd,njd->nkj',vec[c],vec[h]);m=mask[:,None,:]
            mean=(score*m).sum(2)/count[:,None]
            ordered=np.sort(np.where(m,score,-1e9),axis=2)
            maximum=np.where(mask.any(1)[:,None],ordered[:,:,-1],0)
            top3=np.where(ordered[:,:,-3:]>-1e8,ordered[:,:,-3:],0).sum(2)/np.minimum(count,3)[:,None]
            add(label+'_history_mean',mean);add(label+'_history_max',maximum);add(label+'_history_top3',top3)
            add(label+'_history_std',np.sqrt(np.maximum((score**2*m).sum(2)/count[:,None]-mean**2,0)))
            for recent in [1,3,5]:
                add(label+f'_recent{recent}_delta',(score[:,:,-recent:]*m[:,:,-recent:]).sum(2)/mask[:,-recent:].sum(1).clip(1)[:,None]-mean)
            for days in [1.,7.]:
                weight=np.exp(-np.maximum(age,0)/days)*mask
                add(label+f'_decay{days}_delta',(score*weight[:,None,:]).sum(2)/weight.sum(1).clip(1e-8)[:,None]-mean)
            for temperature in [1.,4.]:
                weight=np.exp(temperature*(score-maximum[:,:,None]))*m
                add(label+f'_att{temperature}_delta',(score*weight).sum(2)/weight.sum(2).clip(1e-8)-mean)
            slate=np.einsum('nkd,njd->nkj',vec[c],vec[c]);off=~np.eye(10,dtype=bool)
            peers=slate[:,off].reshape(len(c),10,9)
            add(label+'_peer_mean',peers.mean(2));add(label+'_peer_max',peers.max(2));add(label+'_peer_std',peers.std(2))
        chunks.append(np.stack(cols,1).astype(np.float32))
        if extra_names is None:extra_names=labels
        elif labels!=extra_names:raise ValueError('Feature names changed between chunks.')
    extra=[np.concatenate(chunks)]
    for column in [0,209,221,241,243,244,248,250]:
        value=np.asarray(raw[:,column]).reshape(-1,10)
        mean=np.nanmean(value,axis=1,keepdims=True);maximum=np.nanmax(value,axis=1,keepdims=True);std=np.nanstd(value,axis=1,keepdims=True)
        for label,x in [('mean',np.broadcast_to(mean,value.shape)),('std',np.broadcast_to(std,value.shape)),('relative',value-mean),('to_max',value-maximum)]:
            extra.append(x.reshape(-1,1));extra_names.append(names[column]+'_slate_'+label)
    result=np.concatenate([raw]+extra,axis=1).astype(np.float32)
    return result,names+extra_names
