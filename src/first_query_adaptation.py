"""Fixed gentle fine-tuning of an existing head on first daily user queries."""
import copy
import time
import numpy as np
import torch
from torch import nn
from calibration_model import Calibrator, normalized


def fit(base, datasets, masks, seed=20261029, epochs=8, threads=3, rank_weight=.5,
        checkpoint_path=None, epoch_callback=None):
    torch.set_num_threads(threads); torch.manual_seed(seed); rng=np.random.default_rng(seed)
    n=sum(int(m.sum()) for m in masks)
    x=torch.empty((n*10,len(base['selected'])));offset=torch.empty(n*10);labels=np.empty(n*10,np.float32)
    position=0
    for data,mask in zip(datasets,masks):
        events=np.flatnonzero(mask)
        for start in range(0,len(events),2000):
            rows=(events[start:start+2000,None]*10+np.arange(10)).ravel();raw=data['x'][rows]
            x[position:position+len(rows)]=normalized(raw,base)
            offset[position:position+len(rows)]=torch.from_numpy(raw[:,base['offset_column']].copy())
            labels[position:position+len(rows)]=data['labels'][rows];position+=len(rows)
    target=np.where(labels.reshape(-1,10).sum(1)>0,labels.reshape(-1,10).argmax(1),10)
    net=Calibrator(len(base['selected']),base['width'],base['architecture']);net.load_state_dict(base['state'])
    averaged=copy.deepcopy(net);navg=0;optimizer=torch.optim.AdamW(net.parameters(),lr=.0002,weight_decay=.1)
    curve=[];completed_epoch=0;config=dict(seed=seed,epochs=epochs,threads=threads,rank_weight=rank_weight,events=n)
    if checkpoint_path is not None and checkpoint_path.exists():
        saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False);assert saved['config']==config
        net.load_state_dict(saved['net']);averaged.load_state_dict(saved['average']);optimizer.load_state_dict(saved['optimizer'])
        navg=saved['navg'];curve=saved['curve'];completed_epoch=saved['epoch'];rng.bit_generator.state=saved['numpy_rng'];torch.set_rng_state(saved['torch_rng'])
    for epoch in range(completed_epoch+1,epochs+1):
        started=time.time();net.train();order=rng.permutation(n);losses=[]
        for start in range(0,n,1024):
            events=order[start:start+1024];rows=(events[:,None]*10+np.arange(10)).ravel()
            z=net(x[rows],offset[rows]).reshape(-1,10);allz=torch.cat([z,torch.zeros(len(events),1)],1)
            loss=nn.functional.cross_entropy(allz,torch.tensor(target[events],dtype=torch.long))
            probability=torch.softmax(allz,1)[:,:10].ravel().clamp(1e-6,1-1e-6);logits=torch.logit(probability)
            clicked=torch.tensor(labels[rows],dtype=torch.bool);positive,negative=logits[clicked],logits[~clicked]
            if len(positive) and len(negative):
                sampled=negative[torch.randint(len(negative),(len(positive),32))]
                loss+=rank_weight*nn.functional.softplus(sampled-positive[:,None]).mean()
            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);optimizer.step();losses.append(loss.item())
        if epoch>=max(1,epochs-2):
            navg+=1
            with torch.no_grad():
                for dst,src in zip(averaged.parameters(),net.parameters()):dst.mul_((navg-1)/navg).add_(src,alpha=1/navg)
        entry=dict(epoch=epoch,seconds=time.time()-started,loss=float(np.mean(losses)));curve.append(entry);print(entry,flush=True)
        if checkpoint_path is not None:
            temporary=checkpoint_path.with_suffix('.partial')
            torch.save(dict(config=config,epoch=epoch,net=net.state_dict(),average=averaged.state_dict(),optimizer=optimizer.state_dict(),navg=navg,curve=curve,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),temporary)
            temporary.replace(checkpoint_path)
        if epoch_callback is not None:epoch_callback(epoch)
    result=copy.deepcopy(base);result.update(state=averaged.state_dict(),train_events=n,initial_training_events=base['train_events'],
        seed=seed,epochs=epochs,rank_weight=rank_weight,curve=curve,adapted_first_queries=True,base_frozen=False,
        adaptation_lr=.0002,initial_normalization_preserved=True)
    return result
