"""Additional numerical features for the V5 calibration model."""
import numpy as np
from scipy.special import softmax,logit

def additional(z,state,d,unit,pc):
    cols=[];names=[];n=len(d['user'])
    def add(name,x):
        x=np.asarray(x,np.float32)
        if x.ndim==1:x=np.repeat(x[:,None],10,axis=1)
        cols.append(x.reshape(-1));names.append(name)
    for name,score in [('vi_user',z['user']),('vi_history',z['history']),('vi_mix',.5*z['user']+.5*z['history'])]:
        p=softmax(np.column_stack([score,np.zeros(n)]),axis=1)[:,:10];add(name+'_logit',logit(np.clip(p,1e-7,1-1e-7)));add(name+'_raw',score);add(name+'_event',p.sum(1));add(name+'_std',score.std(1))
    user=d['user'];items=d['items'];um=state['user.weight'].numpy();ul=state['user_logstd.weight'].numpy()[:,0];im=state['item.weight'].numpy();il=state['item_logstd.weight'].numpy()[:,0];un=np.linalg.norm(um,axis=1);inn=np.linalg.norm(im,axis=1);add('vi_user_norm',un[user]);add('vi_user_logstd',ul[user]);add('vi_user_certainty',un[user]/np.sqrt(un[user]**2+7*np.exp(2*ul[user])));add('vi_item_norm',inn[items]);add('vi_item_logstd',il[items]);add('vi_item_certainty',inn[items]/np.sqrt(inn[items]**2+7*np.exp(2*il[items])));add('vi_outside',state['outside.weight'].numpy()[user,0]);score=.75*unit['user']+.25*unit['history'];p=softmax(score,axis=1)*pc;add('cluster_unit_logit',logit(np.clip(p,1e-7,1-1e-7)));add('cluster_unit_user',unit['user']);add('cluster_unit_history',unit['history']);add('cluster_unit_std',score.std(1));return np.stack(cols,1),names
