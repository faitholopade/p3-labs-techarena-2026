"""Smooth event calibration with stable feature selection."""
import copy
import numpy as np
import torch
from torch import nn
from scipy.special import softmax,expit

SELECTED=list(range(31))+list(range(215,307))

class Calibrator(nn.Module):
    def __init__(self,dim,width=32,architecture='plain'):
        super().__init__();self.linear=nn.Linear(dim,1);nn.init.zeros_(self.linear.weight);nn.init.zeros_(self.linear.bias)
        self.mlp=nn.Sequential(nn.Linear(dim,width),nn.SiLU(),nn.Dropout(.1),nn.Linear(width,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,1));nn.init.zeros_(self.mlp[-1].weight);nn.init.zeros_(self.mlp[-1].bias)
        self.architecture=architecture
        if architecture=='set':
            self.event=nn.Sequential(nn.Linear(64,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,1));nn.init.zeros_(self.event[-1].weight);nn.init.zeros_(self.event[-1].bias)
        if architecture=='cross':
            self.down=nn.ModuleList([nn.Linear(dim,16,bias=False) for _ in range(2)]);self.up=nn.ModuleList([nn.Linear(16,dim) for _ in range(2)]);self.cross_out=nn.Linear(dim,1,bias=False)
            for layer in self.up:nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
            nn.init.zeros_(self.cross_out.weight)
    def forward(self,x,offset):
        h=self.mlp[:-1](x);z=(self.linear(x)+self.mlp[-1](h)).squeeze(-1)+offset
        if self.architecture=='set':
            h=h.reshape(-1,10,32);context=torch.cat([h.mean(1),h.max(1).values],1);z=z-self.event(context).repeat_interleave(10).ravel()
        if self.architecture=='cross':
            crossed=x
            for down,up in zip(self.down,self.up):crossed=crossed+x*up(down(crossed))
            z=z+self.cross_out(crossed).squeeze(-1)
        return z

def normalized(raw,spec):
    x=raw[:,spec['selected']];x=np.where(np.isnan(x),spec['median'],x)
    return torch.from_numpy(np.clip((x-spec['mean'])/spec['std'],-spec.get('clip',8),spec.get('clip',8)).astype(np.float32))

def fit(raw,y,names,seed=731,epochs=20,swa_start=12,width=32,mode="event",pos_weight=1.,selection="stable",clip=8.,architecture='plain',rank_weight=0.):
    torch.set_num_threads(3);torch.manual_seed(seed);rng=np.random.default_rng(seed)
    selected=(SELECTED+list(range(307,raw.shape[1]))) if selection=="compact" else [j for j in range(raw.shape[1]) if j not in [32,55,203,205,206]];selected=[j for j in selected if np.nanstd(raw[:,j])>1e-6];x=raw[:,selected];med=np.nanmedian(x,0);x=np.where(np.isnan(x),med,x);mu=x.mean(0);sd=np.maximum(x.std(0),1e-3)
    spec={'selected':selected,'selection':selection,'clip':clip,'architecture':architecture,'rank_weight':rank_weight,'median':med,'mean':mu,'std':sd,'names':names,'seed':seed,'epochs':epochs,'swa_start':swa_start,'width':width,'lr':.001,'weight_decay':.1,'offset_column':224 if mode=='event' else 223,'mode':mode,'pos_weight':pos_weight}
    x=torch.from_numpy(np.clip((x-mu)/sd,-clip,clip).astype(np.float32));o=torch.from_numpy(raw[:,spec['offset_column']].copy()+(np.log(pos_weight) if mode=='binary' else 0));yb=y.reshape(-1,10);target=np.where(yb.sum(1)>0,yb.argmax(1),10);n=len(target);net=Calibrator(len(selected),width,architecture);avg=copy.deepcopy(net);navg=0;opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1);curve=[]
    for epoch in range(1,epochs+1):
        net.train();order=rng.permutation(n);losses=[]
        for st in range(0,n,1024):
            ev=order[st:st+1024];ix=(ev[:,None]*10+np.arange(10)).ravel();z=net(x[ix],o[ix]).reshape(-1,10);loss=nn.functional.cross_entropy(torch.cat([z,torch.zeros(len(ev),1)],1),torch.tensor(target[ev],dtype=torch.long)) if mode=='event' else nn.functional.binary_cross_entropy_with_logits(z.ravel(),torch.tensor(y[ix],dtype=torch.float32),pos_weight=torch.tensor(pos_weight));
            if rank_weight:
                prob=torch.softmax(torch.cat([z,torch.zeros(len(ev),1)],1),1)[:,:10].ravel().clamp(1e-6,1-1e-6);logits=torch.logit(prob);labels=torch.tensor(y[ix],dtype=torch.bool);positive=logits[labels];negative=logits[~labels];sampled=negative[torch.randint(len(negative),(len(positive),32))];loss=loss+rank_weight*nn.functional.softplus(sampled-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        if epoch>=swa_start:
            navg+=1
            with torch.no_grad():
                for dest,source in zip(avg.parameters(),net.parameters()):dest.mul_((navg-1)/navg).add_(source,alpha=1/navg)
        curve.append(float(np.mean(losses)))
    spec['state']=(avg if navg else net).state_dict();spec['loss_curve']=curve
    return spec

@torch.no_grad()
def predict(spec,raw):
    torch.set_num_threads(3);net=Calibrator(len(spec['selected']),spec['width'],spec.get('architecture','plain'));net.load_state_dict(spec['state']);net.eval();parts=[]
    for st in range(0,len(raw),40960):
        r=raw[st:st+40960];x=normalized(r,spec);o=torch.from_numpy(np.asarray(r[:,spec['offset_column']],np.float32).copy());parts.append(net(x,o).numpy())
    z=np.concatenate(parts).reshape(-1,10)
    if spec.get('mode','event')=='binary':return expit(z).ravel()
    return softmax(np.c_[z,np.zeros(len(z))],1)[:,:10].ravel()
