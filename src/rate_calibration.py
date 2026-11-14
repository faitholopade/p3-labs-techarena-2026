"""History calibration with a bounded event-level rate correction."""
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from sequence_calibration import SequenceCalibrator
from calibration_model import normalized


class RateCalibrator(SequenceCalibrator):
    def __init__(self,ck,use_rates=True):
        super().__init__(ck);self.use_rates=use_rates
        self.rate=nn.Sequential(nn.Linear(20,16),nn.SiLU(),nn.Dropout(.1),nn.Linear(16,1))
        nn.init.zeros_(self.rate[-1].weight);nn.init.zeros_(self.rate[-1].bias)

    def forward(self,x,offset,pair,mask,rate):
        z=super().forward(x,offset,pair,mask)
        return z+.5*torch.tanh(self.rate(rate)) if self.use_rates else z


@torch.no_grad()
def predict(ck,raw,pairs,mask,rates):
    torch.set_num_threads(2);net=RateCalibrator(ck['base_spec'],ck['use_rates'])
    net.load_state_dict(ck['state']);net.eval();parts=[]
    for st in range(0,len(pairs),1024):
        sl=slice(st,st+1024);rows=slice(st*10,min(st+1024,len(pairs))*10);r=raw[rows]
        x=normalized(r,ck['base_spec']);off=torch.tensor(r[:,ck['base_spec']['offset_column']].copy())
        rr=torch.tensor(np.clip((rates[sl]-ck['rate_mean'])/ck['rate_std'],-8,8),dtype=torch.float32)
        parts.append(net(x,off,torch.tensor(pairs[sl]),torch.tensor(mask[sl]),rr).numpy())
    z=np.concatenate(parts);return softmax(np.c_[z,np.zeros(len(z))],axis=1)[:,:10].ravel()
