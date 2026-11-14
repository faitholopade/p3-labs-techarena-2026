"""Portable inference for the frozen preference plus exposure-residual model."""
import math,numpy as np,torch
from torch import nn
from scipy.special import softmax,logit
from variational import Variational,predict as preference_predict
from uncertainty import logstd_summary

class Residual(Variational):
    def attach(self,q,mu):
        for p in self.parameters():p.requires_grad_(False)
        self.register_buffer('exposure_items',torch.tensor(q));self.exposure_user=nn.Embedding.from_pretrained(torch.tensor(mu),freeze=False)
        self.exposure_logstd=nn.Embedding.from_pretrained(torch.full((len(mu),1),-.7),freeze=False);self.log_exposure_beta=nn.Parameter(torch.tensor(math.log(3.)));self.item_bias=nn.Embedding.from_pretrained(torch.zeros(len(q),1),freeze=False)
    def forward(self,c,u,sample=True,context=None,preference=None):
        z=super().forward(c,u,sample,context,preference);mu=self.exposure_user(u);std=self.exposure_logstd(u).clamp(-4,1).exp()
        if sample:
            noise=torch.randn_like(mu)
            if sample==-1:noise=-noise
            uv=(mu+std*noise)/math.sqrt(8)
        else:uv=mu/math.sqrt(8)
        beta=self.log_exposure_beta.clamp(math.log(.5),math.log(8)).exp();return z+(self.exposure_items[c]*uv[:,None]).sum(-1)*beta+self.item_bias(c).squeeze(-1)

def load(path):
    ck=torch.load(path,map_location='cpu',weights_only=False);s=ck['state'];v=np.zeros_like(s['item.weight'].numpy());u=np.zeros_like(s['user.weight'].numpy());net=Residual(v,u,np.ones(len(v)),np.ones(len(u)),np.zeros(len(u),np.float32),context_model='context_matrix' in s);net.attach(s['exposure_items'].numpy(),np.zeros_like(s['exposure_user.weight'].numpy()))
    for name in ['item_logstd','user_logstd','exposure_logstd']:
        if getattr(net,name).weight.shape!=s[name+'.weight'].shape:
            setattr(net,name,nn.Embedding.from_pretrained(torch.zeros_like(s[name+'.weight']),freeze=False))
    net.load_state_dict(s);net.eval();return net

@torch.no_grad()
def features(net,d):
    torch.set_num_threads(3);user,history=preference_predict(net,d);base=[]
    for start in range(0,len(d['user']),4096):
        sl=slice(start,start+4096);base.append(Variational.forward(net,torch.tensor(d['items'][sl]),torch.tensor(d['user'][sl]),False,torch.tensor(d['context'][sl])).numpy())
    base=np.concatenate(base);cols=[];names=[];n=len(user)
    def add(name,x):
        x=np.asarray(x,np.float32)
        if x.ndim==1:x=np.repeat(x[:,None],10,1)
        cols.append(x.ravel());names.append(name)
    for name,score in [('rx_user',user),('rx_history',history),('rx_mix',.5*user+.5*history)]:
        p=softmax(np.c_[score,np.zeros(n)],1)[:,:10];add(name+'_raw',score);add(name+'_logit',logit(np.clip(p,1e-7,1-1e-7)));add(name+'_event',p.sum(1));add(name+'_std',score.std(1))
    add('rx_extra',user-base);u=d['user'];c=d['items'];add('rx_extra_user_norm',np.linalg.norm(net.exposure_user.weight.detach().numpy()[u],axis=1));add('rx_extra_user_logstd',logstd_summary(net.exposure_logstd.weight.detach().numpy()[u],-4,1));add('rx_item_bias',net.item_bias.weight.detach().numpy()[c,0]);return np.stack(cols,1),names

