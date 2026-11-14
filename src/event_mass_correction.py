"""Small bounded event-mass residual with conditional item choices frozen."""
import copy
import time
import numpy as np
import torch
from torch import nn
from scipy.special import expit, logit, logsumexp
from bounded_history import base_logits

USER_NAMES = ['user_event_logit', 'history_length_scaled', 'last_click_hours_log',
              'user_events_log', 'hour_bucket', 'hour', 'history_unique', 'history_span_log',
              'history_mean_ctr', 'history_last_ctr', 'new_user_norm', 'new_user_logstd',
              'new_user_certainty', 'new_outside', 'rx_extra_user_norm', 'rx_extra_user_logstd']
EXCLUDED = {'activity_log_observation_days', 'activity_last_day_count', 'activity_last_context_count'}


def feature_names(names):
    return [name for name in names if name in USER_NAMES or name.startswith('history_clicks_')
            or (name.startswith('activity_') and name not in EXCLUDED)]


def features(raw, names, utility):
    selected_names = feature_names(names); selected = [names.index(name) for name in selected_names]
    out = np.empty((len(raw)//10, len(selected)+1), np.float32)
    for start in range(0, len(out), 2048):
        values = raw[start*10:(start+2048)*10][:, selected].reshape(-1,10,len(selected))
        equal = (values == values[:, :1]) | (np.isnan(values) & np.isnan(values[:, :1]))
        assert equal.all(), 'An event feature varies across candidate rows.'
        out[start:start+len(values), :-1] = values[:, 0]
    out[:, -1] = logsumexp(utility, axis=1)
    return out, selected_names + ['frozen_event_logit']


class EventCorrection(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear = nn.Linear(dim, 1)
        self.mlp = nn.Sequential(nn.Linear(dim,16), nn.SiLU(), nn.Dropout(.1), nn.Linear(16,1))
        for layer in [self.linear, self.mlp[-1]]:
            nn.init.zeros_(layer.weight); nn.init.zeros_(layer.bias)

    def forward(self, x):
        return .5 * torch.tanh((self.linear(x) + self.mlp(x)).squeeze(-1))


def fit(base, datasets, masks, seed=20261030, epochs=12, threads=2, checkpoint_path=None,
        epoch_callback=None):
    torch.set_num_threads(threads); torch.manual_seed(seed); rng=np.random.default_rng(seed)
    utility=torch.from_numpy(np.concatenate([d['utility'][m] for d,m in zip(datasets,masks)]))
    raw=np.concatenate([d['event_features'][m] for d,m in zip(datasets,masks)])
    labels=np.concatenate([d['labels'].reshape(-1,10)[m] for d,m in zip(datasets,masks)])
    mean=np.nanmean(raw,axis=0,dtype=np.float64).astype(np.float32)
    raw=np.where(np.isfinite(raw),raw,mean)
    std=np.maximum(raw.std(0,dtype=np.float64),1e-3).astype(np.float32)
    x=torch.from_numpy(np.clip((raw-mean)/std,-8,8));del raw
    target=np.where(labels.sum(1)>0,labels.argmax(1),10)
    net=EventCorrection(x.shape[1]);average=copy.deepcopy(net);navg=0
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.1)
    curve=[];done=0;config=dict(seed=seed,epochs=epochs,threads=threads,events=len(target))
    if checkpoint_path is not None and checkpoint_path.exists():
        ck=torch.load(checkpoint_path,map_location='cpu',weights_only=False);assert ck['config']==config
        net.load_state_dict(ck['net']);average.load_state_dict(ck['average']);opt.load_state_dict(ck['optimizer'])
        curve=ck['curve'];done=ck['epoch'];navg=ck['navg'];rng.bit_generator.state=ck['numpy_rng'];torch.set_rng_state(ck['torch_rng'])
    for epoch in range(done+1,epochs+1):
        start=time.time();net.train();order=rng.permutation(len(target));losses=[]
        for at in range(0,len(target),1024):
            ev=order[at:at+1024];z=utility[ev]+net(x[ev])[:,None]
            allz=torch.cat([z,torch.zeros(len(ev),1)],1)
            loss=nn.functional.cross_entropy(allz,torch.tensor(target[ev],dtype=torch.long))
            probability=torch.softmax(allz,1)[:,:10].ravel().clamp(1e-6,1-1e-6)
            logits=torch.logit(probability);clicked=torch.tensor(labels[ev].ravel(),dtype=torch.bool)
            positive,negative=logits[clicked],logits[~clicked]
            if len(positive) and len(negative):
                sampled=negative[torch.randint(len(negative),(len(positive),32))]
                loss += .5*nn.functional.softplus(sampled-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(net.parameters(),5);opt.step();losses.append(loss.item())
        if epoch >= max(1,epochs-4):
            navg+=1
            with torch.no_grad():
                for dst,src in zip(average.parameters(),net.parameters()):dst.mul_((navg-1)/navg).add_(src,alpha=1/navg)
        rec=dict(epoch=epoch,seconds=time.time()-start,loss=float(np.mean(losses)));curve.append(rec);print(rec,flush=True)
        if checkpoint_path is not None:
            temp=checkpoint_path.with_suffix('.partial')
            torch.save(dict(config=config,net=net.state_dict(),average=average.state_dict(),optimizer=opt.state_dict(),curve=curve,epoch=epoch,navg=navg,numpy_rng=rng.bit_generator.state,torch_rng=torch.get_rng_state()),temp)
            temp.replace(checkpoint_path)
        if epoch_callback is not None:epoch_callback(epoch)
    return dict(state=average.state_dict(),mean=mean,std=std,event_names=datasets[0]['event_names'],
                names=base['names'],base_spec=base,base_frozen=True,train_events=len(target),seed=seed,
                architecture='event_mass',width=16,epochs=epochs,curve=curve)


@torch.no_grad()
def delta(model, event_features):
    net=EventCorrection(len(model['mean']));net.load_state_dict(model['state']);net.eval();parts=[]
    for start in range(0,len(event_features),4096):
        raw=event_features[start:start+4096];raw=np.where(np.isfinite(raw),raw,model['mean'])
        x=np.clip((raw-model['mean'])/model['std'],-8,8).astype(np.float32)
        parts.append(net(torch.from_numpy(x)).numpy())
    return np.concatenate(parts)


def transfer(probability, correction, strength):
    assert 0<=strength<=1 and np.isfinite(correction).all() and np.max(abs(correction))<=.500001
    p=np.asarray(probability,np.float64).reshape(-1,10);mass=p.sum(1)
    assert len(p)==len(correction) and (p>=0).all() and (mass>0).all() and (mass<1).all()
    changed=expit(logit(mass)+strength*correction)
    result=p*(changed/mass)[:,None]
    assert np.isfinite(result).all() and result.min()>=0 and result.sum(1).max()<1
    return result.ravel()
