"""Candidate choice probabilities integrated over latent posterior uncertainty."""
import numpy as np
import torch,math,time,argparse
from torch import nn
from uncertainty import torch_direction

@torch.no_grad()
def features(s,d,samples=64,user_cholesky=None,exact_direction=False):
    torch.set_num_threads(3);engine=torch.quasirandom.SobolEngine(88+s['exposure_user.weight'].shape[1],scramble=True,seed=920);uniform=engine.draw(samples).clamp(1e-6,1-1e-6);noise=torch.erfinv(2*uniform-1)*math.sqrt(2);noise=torch.cat([noise,-noise],0)
    im=s['item.weight'];il=s['item_logstd.weight'].clamp(-3,1).exp();um=s['user.weight'];ul=s['user_logstd.weight'].clamp(-3,1).exp();em=s['exposure_user.weight'];el=s['exposure_logstd.weight'].clamp(-4,1).exp();q=s['exposure_items'];beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp();imean=torch_direction(im,il,exact=exact_direction);parts=[];names=[]
    for start in range(0,len(d['user']),256):
        sl=slice(start,start+256);c=torch.tensor(d['items'][sl]);u=torch.tensor(d['user'][sl]);h=torch.tensor(d['hist'][sl]);mask=(h>0).float();hm=nn.functional.normalize((imean[h]*mask[:,:,None]).sum(1),dim=-1);hc=mask.sum(1,keepdim=True);hm*=torch.sqrt(hc/(hc+10));cols=[];names=[]
        def add(name,value):
            if value.ndim==1:value=value[:,None].expand(-1,10)
            cols.append(value.reshape(-1).numpy());names.append(name)
        joint_noise=None if user_cholesky is None else torch.einsum('nij,sj->nsi',user_cholesky[u],torch.cat([noise[:,:8],noise[:,88:]],1))
        for scale in [.5,1.]:
            user_noise=noise[None,:,:8] if joint_noise is None else joint_noise[:,:,:8]
            exposure_noise=noise[None,:,88:] if joint_noise is None else joint_noise[:,:,8:]
            us=nn.functional.normalize(um[u,None]+scale*ul[u,None]*user_noise,dim=-1)
            iv=nn.functional.normalize(im[c][:,None]+scale*il[c][:,None]*noise[None,:,8:88].reshape(1,len(noise),10,8),dim=-1)
            eu=(em[u,None]+scale*el[u,None]*exposure_noise)/math.sqrt(8);extra=torch.einsum('nkd,nsd->nsk',q[c],eu)*eb+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]
            history=hm[:,None,None]
            if 'context_matrix' in s:
                context=torch.tensor(d['context'][sl]);matrix=s['context_matrix'][context];vector=s['context_vector.weight'][context]
                us=us+torch.einsum('nsd,ndk->nsk',us,matrix)+vector[:,None]
                history=(hm+torch.einsum('nd,ndk->nk',hm,matrix)+vector)[:,None,None]
                extra=extra+s['context_outside.weight'][context,0][:,None,None]
            zu=(iv*us[:,:,None]).sum(-1)*beta+extra;zh=(iv*history).sum(-1)*beta+extra
            for mix in [.5,1.]:
                score=mix*zu+(1-mix)*zh;prob=torch.softmax(torch.cat([score,torch.zeros(len(u),len(noise),1)],2),2);mean=prob.mean(1);p=mean[:,:10].clamp(1e-8,1-1e-8);prefix=f'posterior{scale}_mix{mix}_'
                add(prefix+'logit',torch.logit(p));add(prefix+'utility',torch.log(p/mean[:,10:].clamp_min(1e-8)));add(prefix+'event',p.sum(1));add(prefix+'prob_std',prob[:,:,:10].std(1));add(prefix+'utility_std',score.std(1))
        parts.append(np.stack(cols,1))
    return np.concatenate(parts).astype(np.float32),names

