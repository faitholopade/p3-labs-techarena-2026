"""Train history calibration with a bounded event-level intensity correction."""
import copy,gc,time
import numpy as np
import torch
from torch import nn
from rate_calibration import RateCalibrator
from calibration_model import normalized

def fit(ck,ds,masks,seed,epochs,use_rates):
    torch.set_num_threads(2);torch.manual_seed(seed+5801);rng=np.random.default_rng(seed+5801)
    ne=sum(int(m.sum()) for m in masks);x=torch.empty((ne*10,len(ck['selected'])))
    off=torch.empty(ne*10);label=np.empty(ne*10,np.float32)
    pair=torch.empty((ne,10,20,6),dtype=torch.float16);hm=torch.empty((ne,20),dtype=torch.bool)
    rates=np.concatenate([d['rates'][m] for d,m in zip(ds,masks)])
    rate_mean=rates.mean(0);rate_std=np.maximum(rates.std(0),.1)
    rates=torch.tensor(np.clip((rates-rate_mean)/rate_std,-8,8),dtype=torch.float32)
    pos=0
    for d,m in zip(ds,masks):
        ix=np.flatnonzero(m)
        for st in range(0,len(ix),2000):
            ev=ix[st:st+2000];rows=(ev[:,None]*10+np.arange(10)).ravel();raw=d['x'][rows]
            dst=slice(pos*10,(pos+len(ev))*10)
            x[dst]=normalized(raw,ck);off[dst]=torch.tensor(raw[:,ck['offset_column']].copy());label[dst]=d['labels'][rows]
            pair[pos:pos+len(ev)]=torch.tensor(d['pairs'][ev]);hm[pos:pos+len(ev)]=torch.tensor(d['mask'][ev]);pos+=len(ev)
    target=np.where(label.reshape(-1,10).sum(1)>0,label.reshape(-1,10).argmax(1),10)
    del raw;gc.collect();net=RateCalibrator(ck,use_rates);avg=copy.deepcopy(net);navg=0
    opt=torch.optim.AdamW([{'params':net.base.parameters(),'lr':.0002},
        {'params':list(net.attention.parameters())+list(net.residual.parameters())+list(net.rate.parameters()),'lr':.002}],weight_decay=.1)
    curve=[]
    for epoch in range(1,epochs+1):
        start=time.time();net.train();order=rng.permutation(ne);losses=[]
        for st in range(0,ne,512):
            ev=order[st:st+512];rows=(ev[:,None]*10+np.arange(10)).ravel()
            z=net(x[rows],off[rows],pair[ev],hm[ev],rates[ev]);allz=torch.cat([z,torch.zeros(len(ev),1)],1)
            loss=nn.functional.cross_entropy(allz,torch.tensor(target[ev]))
            prob=torch.softmax(allz,1)[:,:10].ravel().clamp(1e-6,1-1e-6);logits=torch.logit(prob)
            lab=torch.tensor(label[rows],dtype=torch.bool);positive=logits[lab];negative=logits[~lab]
            if len(positive):loss+=nn.functional.softplus(negative[torch.randint(len(negative),(len(positive),32))]-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        if epoch>=max(1,epochs-2):
            navg+=1
            with torch.no_grad():
                for dst,src in zip(avg.parameters(),net.parameters()):dst.mul_((navg-1)/navg).add_(src,alpha=1/navg)
        rec={'epoch':epoch,'seconds':time.time()-start,'loss':float(np.mean(losses))};curve.append(rec);print(rec,flush=True)
    return {'base_spec':ck,'state':avg.state_dict(),'seed':seed+5801,'epochs':epochs,'training_events':ne,'curve':curve,'rate_mean':rate_mean,'rate_std':rate_std,'use_rates':use_rates}

