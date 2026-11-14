"""Portable smooth calibration, with explicit preprocessing and parameter averaging."""
import copy
import numpy as np
import torch
from torch import nn
from scipy.special import softmax,expit

SELECTED=list(range(31))+list(range(215,307))

class Calibrator(nn.Module):
    def __init__(self,dim,width=32):
        super().__init__();self.linear=nn.Linear(dim,1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
        self.mlp=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Dropout(.1),nn.Linear(width,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,1));nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
    def forward(self,x,offset):return (self.linear(x)+self.mlp(x)).squeeze(-1)+offset

def normalized(raw,spec):
    x=raw[:,spec['selected']];x=np.where(np.isnan(x),spec['median'],x)
    return torch.from_numpy(np.clip((x-spec['mean'])/spec['std'],-8,8).astype(np.float32))

def fit(raw,y,names,seed=731,epochs=20,swa_start=12,width=32,mode="event",pos_weight=1.):
    torch.set_num_threads(3);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    selected=SELECTED+list(range(307,raw.shape[1]));x=raw[:,selected];med=np.nanmedian(x,0);x=np.where(np.isnan(x),med,x);mu=x.mean(0);sd=np.maximum(x.std(0),1e-3)
    spec={'selected':selected,'median':med,'mean':mu,'std':sd,'names':names,'seed':seed,'epochs':epochs,'swa_start':swa_start,'width':width,'lr':.001,'weight_decay':.1,'offset_column':224 if mode=='event' else 223,'mode':mode,'pos_weight':pos_weight}
    x=torch.from_numpy(np.clip((x-mu)/sd,-8,8).astype(np.float32));o=torch.from_numpy(raw[:,spec['offset_column']].copy()+(np.log(pos_weight) if mode=='binary' else 0));yb=y.reshape(-1,10);target=np.where(yb.sum(1)>0,yb.argmax(1),10);n=len(target);net=Calibrator(len(selected),width);avg=copy.deepcopy(net);navg=0;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1);curve=[]
    for epoch in range(1,epochs+1):
        net.train();order=rng.permutation(n);losses=[]
        for st in range(0,n,1024):
            ev=order[st:st+1024];ix=(ev[:,None]*10+np.arange(10)).ravel();z=net(x[ix],o[ix]).reshape(-1,10);loss=nn.functional.cross_entropy(torch.cat([z,torch.zeros(len(ev),1)],1),torch.tensor(target[ev],dtype=torch.long)) if mode=='event' else nn.functional.binary_cross_entropy_with_logits(z.ravel(),torch.tensor(y[ix],dtype=torch.float32),pos_weight=torch.tensor(pos_weight));opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        if epoch>=swa_start:
            navg+=1
            with torch.no_grad():
                for dest,source in zip(avg.parameters(),net.parameters()):dest.mul_((navg-1)/navg).add_(source,alpha=1/navg)
        curve.append(float(np.mean(losses)))
    spec['state']=(avg if navg else net).state_dict();spec['loss_curve']=curve
    return spec

@torch.no_grad()
def predict(spec,raw):
    torch.set_num_threads(3);net=Calibrator(len(spec['selected']),spec['width']);net.load_state_dict(spec['state']);net.eval();parts=[]
    for st in range(0,len(raw),40960):
        r=raw[st:st+40960];x=normalized(r,spec);o=torch.from_numpy(np.asarray(r[:,spec['offset_column']],np.float32).copy());parts.append(net(x,o).numpy())
    z=np.concatenate(parts).reshape(-1,10)
    if spec.get('mode','event')=='binary':return expit(z).ravel()
    return softmax(np.c_[z,np.zeros(len(z))],1)[:,:10].ravel()
