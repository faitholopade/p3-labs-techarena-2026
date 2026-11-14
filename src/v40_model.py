"""Permutation-equivariant candidate corrections conditioned on the full slate."""
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from calibration_model import Calibrator as SharedSlateCalibrator,normalized

class Calibrator(SharedSlateCalibrator):
    def __init__(self,dim,width=32,architecture='candidate_set'):
        if architecture!='candidate_set':raise ValueError('Expected candidate_set architecture')
        super().__init__(dim,width,'set')
        self.architecture=architecture
        self.candidate_context=nn.Sequential(nn.Linear(96,32),nn.SiLU(),nn.Dropout(.1),nn.Linear(32,1))
        nn.init.zeros_(self.candidate_context[-1].weight)
        nn.init.zeros_(self.candidate_context[-1].bias)

    def forward(self,x,offset):
        h=self.mlp[:-1](x);event_h=h.reshape(-1,10,32)
        summary=torch.cat([event_h.mean(1),event_h.max(1).values],1)
        context=summary[:,None,:].expand(-1,10,-1)
        correction=self.candidate_context(torch.cat([event_h,context],2)).squeeze(-1)
        correction=correction-correction.mean(1,keepdim=True)
        base=(self.linear(x)+self.mlp[-1](h)).squeeze(-1)+offset
        return base+correction.ravel()-self.event(summary).repeat_interleave(10).ravel()

@torch.no_grad()
def predict(spec,raw):
    torch.set_num_threads(2)
    net=Calibrator(len(spec['selected']),spec['width'],spec['architecture'])
    net.load_state_dict(spec['state']);net.eval();parts=[]
    for start in range(0,len(raw),40960):
        block=raw[start:start+40960];x=normalized(block,spec)
        offset=torch.from_numpy(np.asarray(block[:,spec['offset_column']],np.float32).copy())
        parts.append(net(x,offset).numpy())
    z=np.concatenate(parts).reshape(-1,10)
    return softmax(np.c_[z,np.zeros(len(z))],1)[:,:10].ravel()
