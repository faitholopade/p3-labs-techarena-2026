"""V40 candidate-conditioned slate training; fixed inherited optimization loop."""
import numpy as np,torch,copy,time,gc
from torch import nn
from v40_model import Calibrator

def train(ds,masks,seed,epochs=20,width=64,offset_name='rx_mix_raw',rank_weight=1.,threads=3,architecture='plain',sample_weights=None,sampling_weights=None):
    torch.set_num_threads(threads);torch.manual_seed(seed);rng=np.random.default_rng(seed);names=ds[0]['names'].tolist()
    selected=list(range(70))+list(range(140,len(names)))
    bad={'item_context_delta','item_context_impressions_log','activity_log_observation_days','activity_last_day_count','activity_last_context_count'}
    selected=[i for i in selected if names[i] not in bad]
    p=len(selected);total=np.zeros(p);sq=np.zeros(p);counts=np.zeros(p);n=sum(int(m.sum()) for m in masks)*10
    for data,mask in zip(ds,masks):
        ix=np.flatnonzero(mask)
        for st in range(0,len(ix),2000):
            r=(ix[st:st+2000,None]*10+np.arange(10)).ravel();a=data['x'][r][:,selected].astype(float);finite=np.isfinite(a);counts+=finite.sum(0);a[~finite]=0;total+=a.sum(0);sq+=(a*a).sum(0)
    mu=total/np.maximum(counts,1);sd=np.sqrt(np.maximum(sq/np.maximum(counts,1)-mu**2,0));keep=sd>1e-6;selected=np.array(selected)[keep].tolist();mu=mu[keep].astype(np.float32);sd=np.maximum(sd[keep],1e-3).astype(np.float32);ncol=len(selected)
    x=torch.empty((n,ncol));offset=torch.empty(n);labels=np.empty(n,np.float32);pos=0;offset_column=names.index(offset_name)
    for data,mask in zip(ds,masks):
        ix=np.flatnonzero(mask)
        for st in range(0,len(ix),2000):
            r=(ix[st:st+2000,None]*10+np.arange(10)).ravel();raw=data['x'][r];a=raw[:,selected];a=np.where(np.isfinite(a),a,mu);x[pos:pos+len(r)]=torch.from_numpy(np.clip((a-mu)/sd,-8,8));offset[pos:pos+len(r)]=torch.from_numpy(raw[:,offset_column].copy());labels[pos:pos+len(r)]=data['labels'][r];pos+=len(r)
    del a,raw;gc.collect();target=np.where(labels.reshape(-1,10).sum(1)>0,labels.reshape(-1,10).argmax(1),10);nevent=len(target);net=Calibrator(ncol,width,architecture);avg=copy.deepcopy(net);navg=0;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1);curve=[]
    weights=None
    sampling_probability=None
    if sampling_weights is not None:
        if sample_weights is not None:raise ValueError('Use sampling or loss weights, not both.')
        sampling_probability=np.concatenate([np.asarray(w,dtype=np.float64)[mask] for w,mask in zip(sampling_weights,masks)])
        if len(sampling_probability)!=nevent or not np.isfinite(sampling_probability).all() or np.any(sampling_probability<=0):raise ValueError('Invalid event sampling weights.')
        sampling_probability/=sampling_probability.sum()
    if sample_weights is not None:
        weights=np.concatenate([np.asarray(w,dtype=np.float32)[mask] for w,mask in zip(sample_weights,masks)])
        if len(weights)!=nevent or not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid event weights.')
        weights=torch.from_numpy(weights/weights.mean())
    print({'training_events':nevent,'features':ncol,'seed':seed,'weighted':weights is not None},flush=True)
    for epoch in range(1,epochs+1):
        start=time.time();net.train();order=rng.permutation(nevent) if sampling_probability is None else rng.choice(nevent,size=nevent,replace=True,p=sampling_probability);losses=[]
        for st in range(0,nevent,1024):
            ev=order[st:st+1024];ix=(ev[:,None]*10+np.arange(10)).ravel();z=net(x[ix],offset[ix]).reshape(-1,10);allz=torch.cat([z,torch.zeros(len(ev),1)],1)
            target_batch=torch.tensor(target[ev],dtype=torch.long)
            loss=nn.functional.cross_entropy(allz,target_batch) if weights is None else (nn.functional.cross_entropy(allz,target_batch,reduction='none')*weights[ev]).mean()
            if rank_weight:
                prob=torch.softmax(allz,1)[:,:10].ravel().clamp(1e-6,1-1e-6);logits=torch.logit(prob);label=torch.tensor(labels[ix],dtype=torch.bool);positive=logits[label];negative=logits[~label]
                if len(positive) and len(negative):
                    sampled=negative[torch.randint(len(negative),(len(positive),32))];loss+=rank_weight*nn.functional.softplus(sampled-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        if epoch>=max(1,epochs-8):
            navg+=1
            with torch.no_grad():
                for dst,src in zip(avg.parameters(),net.parameters()):dst.mul_((navg-1)/navg).add_(src,alpha=1/navg)
        curve.append(float(np.mean(losses)))
        if epoch%5==0:print({'epoch':epoch,'loss':curve[-1],'seconds':time.time()-start},flush=True)
    return {'state':avg.state_dict(),'names':names,'selected':selected,'median':mu,'mean':mu,'std':sd,'clip':8,'architecture':architecture,'width':width,'offset_column':offset_column,'mode':'event','seed':seed,'epochs':epochs,'rank_weight':rank_weight,'train_events':nevent,'loss_curve':curve}

