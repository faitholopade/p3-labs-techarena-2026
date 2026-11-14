"""A bounded candidate-aware history correction to a saved event calibrator."""
import copy,math
import numpy as np
import torch
from torch import nn
from scipy.special import softmax
from calibration_model import Calibrator,normalized
from uncertainty import numpy_direction


def history_pairs(state,d,exact_direction=False):
    mu=state['item.weight'].numpy();sd=state['item_logstd.weight'].clamp(-3,1).exp().numpy()
    v=numpy_direction(mu,sd,exact=exact_direction)
    v[0]=0;certainty=np.linalg.norm(v,axis=1)
    # Keep the existing history-feature geometry when adding utility factors.
    q=state['exposure_items'].numpy()[:,:8];q=q.copy();q[0]=0
    out=np.empty((len(d['user']),10,20,6),np.float16)
    mask=d['hist']>0
    assert np.all(d['hist_ts'][mask]<np.broadcast_to(d['timestamp'][:,None],mask.shape)[mask])
    for st in range(0,len(out),1024):
        sl=slice(st,st+1024);c=d['items'][sl];h=d['hist'][sl];t=d['timestamp'][sl,None];ht=d['hist_ts'][sl]
        out[sl,:,:,0]=4*np.einsum('nkd,njd->nkj',v[c],v[h])
        out[sl,:,:,1]=2*np.einsum('nkd,njd->nkj',q[c],q[h])
        out[sl,:,:,2]=np.minimum(np.log1p(np.maximum(t-ht,0)/86400)/4,4)[:,None,:]
        out[sl,:,:,3]=np.arange(-19,1,dtype=np.float32)[None,None,:]/20
        out[sl,:,:,4]=certainty[h][:,None,:]
        out[sl,:,:,5]=np.cos(2*np.pi*(t-ht)/86400)[:,None,:]
    return out,mask


class SequenceCalibrator(nn.Module):
    def __init__(self,ck):
        super().__init__();self.base=Calibrator(len(ck['selected']),ck['width'],ck.get('architecture','plain'))
        self.base.load_state_dict(ck['state'])
        self.attention=nn.Sequential(nn.Linear(6,12),nn.SiLU(),nn.Linear(12,1))
        self.residual=nn.Sequential(nn.Linear(14,16),nn.SiLU(),nn.Dropout(.1),nn.Linear(16,1))
        for layer in [self.attention[-1],self.residual[-1]]:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,x,offset,pair,mask):
        pair=pair.float();m=mask[:,None,:];n=m.sum(2).clamp_min(1)
        att=self.attention(pair).squeeze(-1).masked_fill(~m,-1e9)
        weight=torch.softmax(att,2)*m;summary=[]
        for k in [0,1]:
            score=pair[:,:,:,k];mean=(score*m).sum(2)/n
            maximum=score.masked_fill(~m,-1e9).max(2).values
            maximum=torch.where(m.any(2),maximum,torch.zeros_like(maximum))
            recent=(score[:,:,-3:]*m[:,:,-3:]).sum(2)/m[:,:,-3:].sum(2).clamp_min(1)
            summary.extend([mean,(score*weight).sum(2)-mean,maximum,
                ((score.square()*m).sum(2)/n-mean.square()).clamp_min(0),
                recent-mean,score[:,:,-1]*m[:,:,-1]-mean])
        summary.append(mask.sum(1)[:,None].expand(-1,10)/20)
        summary.append((pair[:,:,:,2]*m).sum(2)/n)
        correction=.5*torch.tanh(self.residual(torch.stack(summary,-1)).squeeze(-1))
        return self.base(x,offset).reshape(-1,10)+correction


@torch.no_grad()
def predict(ck,raw,pairs,mask):
    torch.set_num_threads(2);model=SequenceCalibrator(ck['base_spec'])
    model.load_state_dict(ck['state']);model.eval();parts=[]
    for st in range(0,len(pairs),1024):
        sl=slice(st,st+1024);rows=slice(st*10,min(st+1024,len(pairs))*10)
        r=raw[rows];x=normalized(r,ck['base_spec'])
        off=torch.tensor(r[:,ck['base_spec']['offset_column']].copy())
        parts.append(model(x,off,torch.tensor(pairs[sl]),torch.tensor(mask[sl])).numpy())
    z=np.concatenate(parts);return softmax(np.c_[z,np.zeros(len(z))],axis=1)[:,:10].ravel()

