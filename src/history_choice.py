"""History-only conditional choice from chronological frozen item representations."""
import copy,math,time
import numpy as np,torch
from torch import nn
from scipy.special import softmax
from uncertainty import numpy_direction


def item_state(state):
    v=numpy_direction(state['item.weight'].numpy(),state['item_logstd.weight'].clamp(-3,1).exp().numpy());v[0]=0
    q=state['exposure_items'].numpy()[:,:8].copy();q[0]=0
    return {'preference':v,'exposure':q,'certainty':np.linalg.norm(v,axis=1),'bias':state['item_bias.weight'].numpy()[:,0]}


def features(data,items):
    c=data['items'];h=data['hist'];ts=data['timestamp'];ht=data['hist_ts'];mask=h>0
    if not np.all(ht[mask]<np.broadcast_to(ts[:,None],ht.shape)[mask]):raise ValueError('History includes query/future events.')
    count=mask.sum(1).clip(1);age=np.maximum(ts[:,None]-ht,0)/86400;columns=[];names=[]
    def add(name,value):
        a=np.asarray(value,np.float32)
        if a.ndim==1:a=np.broadcast_to(a[:,None],c.shape)
        columns.append(a);names.append(name)
    add('history_count',mask.sum(1)/20);add('last_age_log',np.log1p(np.where(mask.any(1),age[:,-1],0)))
    add('mean_age_log',(np.log1p(age)*mask).sum(1)/count)
    add('clock_sin',np.sin(2*np.pi*(ts%86400)/86400));add('clock_cos',np.cos(2*np.pi*(ts%86400)/86400))
    same_hour=(ht%86400//21600==ts[:,None]%86400//21600)&mask
    same_day=(((ht//86400+3)%7>=5)==((ts[:,None]//86400+3)%7>=5))&mask
    add('same_hour_support',same_hour.sum(1)/20);add('same_day_support',same_day.sum(1)/20)
    base=None
    for label in ['preference','exposure']:
        vec=items[label];score=np.einsum('nkd,njd->nkj',vec[c],vec[h]);m=mask[:,None,:]
        mean=(score*m).sum(2)/count[:,None];hv=(vec[h]*mask[:,:,None]).sum(1)/count[:,None];norm=np.linalg.norm(hv,axis=1)
        ordered=np.sort(np.where(m,score,-1e6),axis=2);maximum=np.where(mask.any(1)[:,None],ordered[:,:,-1],0)
        top3=np.where(ordered[:,:,-3:]>-1e5,ordered[:,:,-3:],0).sum(2)/np.minimum(count,3)[:,None]
        add(label+'_mean',mean);add(label+'_cosine',mean/np.maximum(norm[:,None],.05));add(label+'_history_norm',norm)
        add(label+'_std',np.sqrt(np.maximum((score**2*m).sum(2)/count[:,None]-mean**2,0)))
        add(label+'_max',maximum);add(label+'_top3',top3)
        for recent in [1,3,5]:add(label+f'_recent{recent}',(score[:,:,-recent:]*m[:,:,-recent:]).sum(2)/mask[:,-recent:].sum(1).clip(1)[:,None])
        for key,weight in [('decay1',np.exp(-age)*mask),('decay7',np.exp(-age/7)*mask),('same_hour',same_hour),('same_day',same_day)]:
            add(label+'_'+key,(score*weight[:,None,:]).sum(2)/weight.sum(1).clip(1e-8)[:,None])
        for temperature in [1.,4.]:
            w=np.exp(temperature*(score-maximum[:,:,None]))*m;add(label+f'_attention{temperature}',(score*w).sum(2)/w.sum(2).clip(1e-8))
        peers=np.einsum('nkd,njd->nkj',vec[c],vec[c]);peer=peers[:,~np.eye(10,dtype=bool)].reshape(len(c),10,9)
        add(label+'_peer_mean',peer.mean(2));add(label+'_peer_max',peer.max(2));add(label+'_peer_std',peer.std(2))
        add(label+'_relative',mean-mean.mean(1,keepdims=True));add(label+'_to_max',mean-mean.max(1,keepdims=True))
        if label=='preference':base=4*mean/np.maximum(norm[:,None],.05)*np.sqrt(count[:,None]/(count[:,None]+10))
    bias=items['bias'][c];add('item_bias',bias);add('item_bias_relative',bias-bias.mean(1,keepdims=True))
    add('candidate_certainty',items['certainty'][c]);add('history_certainty',(items['certainty'][h]*mask).sum(1)/count)
    add('repeat_count',((c[:,:,None]==h[:,None,:])&mask[:,None,:]).sum(2));add('is_last',(c==h[:,-1,None])&mask[:,-1,None])
    result=np.stack(columns,2).astype(np.float32);base=base+bias
    if not np.isfinite(result).all() or not np.isfinite(base).all():raise ValueError('Nonfinite history features.')
    return result,base.astype(np.float32),names


class HistoryChoice(nn.Module):
    def __init__(self,dimension,width=64):
        super().__init__();self.linear=nn.Linear(dimension,1);self.net=nn.Sequential(nn.Linear(dimension,width),nn.SiLU(),nn.Dropout(.1),nn.Linear(width,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,1))
        for m in [self.linear,self.net[-1]]:nn.init.zeros_(m.weight);nn.init.zeros_(m.bias)
    def forward(self,x,offset):return offset+(self.linear(x)+self.net(x)).squeeze(-1)


def train(raw,offset,target,names,seed=20261111,epochs=16,width=64,threads=2):
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(seed);n=len(target)
    mean=np.zeros(raw.shape[2]);sq=mean.copy()
    for start in range(0,n,4000):
        block=np.asarray(raw[start:start+4000],np.float64);mean+=block.sum((0,1));sq+=(block*block).sum((0,1))
    mean/=n*10;std=np.sqrt(np.maximum(sq/(n*10)-mean**2,1e-6));mean=mean.astype(np.float32);std=std.astype(np.float32)
    model=HistoryChoice(len(names),width);avg=copy.deepcopy(model);navg=0;opt=torch.optim.AdamW(model.parameters(),lr=.002,weight_decay=.05);records=[]
    for epoch in range(1,epochs+1):
        started=time.time();model.train();order=rng.permutation(n);losses=[]
        for st in range(0,n,1024):
            ix=order[st:st+1024];x=torch.from_numpy(np.clip((np.asarray(raw[ix],np.float32)-mean)/std,-8,8));off=torch.from_numpy(np.asarray(offset[ix],np.float32))
            logits=model(x,off);loss=nn.functional.cross_entropy(logits,torch.tensor(target[ix],dtype=torch.long))
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();losses.append(float(loss.detach()))
        if epoch>=8:
            navg+=1
            with torch.no_grad():
                for dst,src in zip(avg.parameters(),model.parameters()):dst.mul_((navg-1)/navg).add_(src,alpha=1/navg)
        rec={'epoch':epoch,'loss':float(np.mean(losses)),'seconds':time.time()-started};records.append(rec);print(rec,flush=True)
    return {'state':avg.state_dict(),'names':names,'mean':mean,'std':std,'clip':8,'width':width,'seed':seed,'epochs':epochs,'training_clicked_events':n,'curve':records,
            'scope':'Conditional candidate choice only; trained on prior clicked slates, no user ID or fitted user-factor inputs.'}


@torch.no_grad()
def predict(model,query,items,threads=2):
    torch.set_num_threads(threads);net=HistoryChoice(len(model['names']),model['width']);net.load_state_dict(model['state']);net.eval();out=[]
    for st in range(0,len(query['items']),2048):
        data={k:v[st:st+2048] for k,v in query.items()};x,off,names=features(data,items)
        if names!=model['names']:raise ValueError('Feature schema mismatch.')
        x=torch.from_numpy(np.clip((x-model['mean'])/model['std'],-model['clip'],model['clip']));out.append(softmax(net(x,torch.from_numpy(off)).numpy(),1))
    return np.concatenate(out).astype(np.float64)
