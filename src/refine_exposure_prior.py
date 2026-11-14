"""Chronological factor refinement with an exposure-informed user prior.

The prior uses only items shown before the training cutoff. It predicts the
existing exposure-residual coefficients from those impression profiles.
"""
import argparse
import hashlib
import json
import math
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from common import load_data, split_data, subset
from residual_model import load
from cluster_unit import warmup
from negative import pooled, sample as sample_pool


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True)
    ap.add_argument('--init', required=True)
    ap.add_argument('--exposure', required=True)
    ap.add_argument('--cut', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool-cache', required=True)
    ap.add_argument('--variance', type=float, default=1.)
    ap.add_argument('--epochs', type=int, default=16)
    ap.add_argument('--warm-weight', type=float, default=1.)
    ap.add_argument('--rank-weight', type=float, default=0.)
    ap.add_argument('--context', choices=['none','vector','full'], default='none')
    ap.add_argument('--diagonal-uncertainty',choices=['none','user','all'],default='none')
    ap.add_argument('--lr',type=float,default=.001)
    ap.add_argument('--exposure-blocks',type=int,choices=[1,2],default=1)
    ap.add_argument('--antithetic',action='store_true')
    ap.add_argument('--new-exposure-logstd',type=float,default=-3.)
    ap.add_argument('--exposure-item-penalty',type=float,default=0.,help='Positive value jointly fits exposure item vectors with a squared-distance anchor to their starting values.')
    ap.add_argument('--warm-density',type=float,default=0.,help='Mix conditional click-density and sampled-choice warm-history losses; value in [0,1].')
    ap.add_argument('--marginal-real',action='store_true',help='Use log mean predictive event likelihood across paired latent draws.')
    ap.add_argument('--users-only',action='store_true',help='Freeze catalog vectors and global coefficients during user posterior refinement.')
    ap.add_argument('--latent-pairs',type=int,default=1,help='Independent antithetic latent pairs per batch; requires antithetic for values above one.')
    ap.add_argument('--prior-file',help='Optional cross-fitted exposure prior means with matching cutoff and source-factor hash.')
    ap.add_argument('--preference-kl',type=float,default=.25,help='Preference and outside-option KL weight; legacy default is 0.25.')
    ap.add_argument('--seed', type=int, default=20261027)
    ap.add_argument('--threads', type=int, default=2)
    a = ap.parse_args()
    if a.variance <= 0:
        raise ValueError('variance must be positive')
    if a.preference_kl<=0:raise ValueError('preference-kl must be positive')
    if not 0<=a.warm_density<=1:raise ValueError('warm-density must be in [0,1].')
    if a.marginal_real and not a.antithetic:raise ValueError('marginal-real requires paired latent draws.')
    if a.users_only and a.exposure_item_penalty>0:raise ValueError('users-only cannot update catalog parameters.')
    if a.latent_pairs<1 or (a.latent_pairs>1 and not a.antithetic):raise ValueError('Multiple latent pairs require antithetic sampling.')
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    data, _, audit = load_data(Path(a.root) / 'data')
    fi, _ = split_data(data, a.cut); fit = subset(data, fi)
    del data
    net = load(a.init)
    if a.exposure_blocks==2:
        if net.exposure_items.shape[1]!=8:raise ValueError('Exposure expansion expects an eight-dimensional starting block.')
        q=np.load(a.exposure)['v'][:,9:17].copy();q/=np.maximum(np.linalg.norm(q,axis=1,keepdims=True),1e-8);q[0]=0
        net.exposure_items=torch.cat([net.exposure_items,torch.tensor(q)],1)
        mu=net.exposure_user.weight.detach();net.exposure_user=nn.Embedding.from_pretrained(torch.cat([mu,torch.zeros_like(mu)],1),freeze=False)
        std=net.exposure_logstd.weight.detach()
        if std.shape[1]==1:std=std.repeat(1,8)
        net.exposure_logstd=nn.Embedding.from_pretrained(torch.cat([std,torch.full_like(std,a.new_exposure_logstd)],1),freeze=False)
    if a.diagonal_uncertainty!='none':
        kinds=['user_logstd','exposure_logstd']+(['item_logstd'] if a.diagonal_uncertainty=='all' else [])
        for kind in kinds:
            weight=getattr(net,kind).weight.detach()
            dimension=net.exposure_user.weight.shape[-1] if kind=='exposure_logstd' else net.dim
            if weight.shape[-1]==1:setattr(net,kind,nn.Embedding.from_pretrained(weight.repeat(1,dimension),freeze=False))
    if a.context!='none':
        net.context_matrix=nn.Parameter(torch.zeros(8,net.dim,net.dim))
        net.context_vector=nn.Embedding(8,net.dim);nn.init.zeros_(net.context_vector.weight)
        net.context_outside=nn.Embedding(8,1);nn.init.zeros_(net.context_outside.weight)
    for param in net.parameters(): param.requires_grad_(True)
    if a.context=='vector':net.context_matrix.requires_grad_(False)
    if a.users_only:
        for name,param in net.named_parameters():
            param.requires_grad_(name.split('.')[0] in ['user','user_logstd','exposure_user','exposure_logstd','outside'])
    exposure_item_anchor=net.exposure_items.detach().clone()
    if a.exposure_item_penalty>0:
        del net._buffers['exposure_items']
        net.exposure_items=nn.Parameter(exposure_item_anchor.clone())
    nu, ni = audit['n_users'] + 1, audit['n_items'] + 1
    count = np.bincount(fit['user'], minlength=nu)
    impressions = sp.coo_matrix((np.ones(len(fi)*10, np.float32),
        (np.repeat(fit['user'],10),fit['items'].ravel())), shape=(nu,ni)).tocsr()
    profile = np.asarray(impressions @ exposure_item_anchor.numpy()) / np.maximum(10*count[:,None],1)
    dense = np.flatnonzero(count >= 10)
    train_users, check_users = train_test_split(dense, test_size=.2, random_state=1729)
    target = net.exposure_user.weight.detach().numpy()
    diagnostic = Ridge(alpha=10).fit(profile[train_users], target[train_users])
    active=np.std(target[dense],axis=0)>1e-8
    prior_report = {'dense_users':len(dense), 'diagnostic_users':len(check_users),
        'heldout_user_r2':float(r2_score(target[check_users][:,active], diagnostic.predict(profile[check_users])[:,active])),
        'nonconstant_diagnostic_dimensions':int(active.sum()),
        'diagnostic_scope':'Predicts pre-cutoff fitted factors; not a CTR validation score.'}
    ridge = Ridge(alpha=10).fit(profile[dense], target[dense])
    prior_mean = torch.tensor(ridge.predict(profile), dtype=torch.float32)
    if a.prior_file:
        external=np.load(a.prior_file)
        with open(a.init,'rb') as stream:initial_sha=hashlib.file_digest(stream,'sha256').hexdigest()
        if int(external['cut'])!=a.cut or str(external['source_factor_sha256'])!=initial_sha:raise ValueError('External prior provenance mismatch.')
        mean=external['mean']
        if mean.shape!=prior_mean.shape or not np.isfinite(mean).all():raise ValueError('Invalid external prior means.')
        prior_mean=torch.tensor(mean,dtype=torch.float32)
        prior_report['external_prior_file']=a.prior_file
        prior_report['external_prior_source_sha256']=initial_sha
    wu, wi, wx = warmup(fit); total = len(fi)+len(wu)
    pool = pooled(fit, ni-1, nu-1, a.cut, basis=a.exposure, cache=a.pool_cache)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    records = []
    print({'prior':prior_report,'events':len(fi),'warmup':len(wu)},flush=True)
    for epoch in range(1,a.epochs+1):
        start=time.time(); net.train()
        opt.param_groups[0]['lr']=a.lr*(.2+.8*.5*(1+math.cos(math.pi*(epoch-1)/max(1,a.epochs-1))))
        order=rng.permutation(total); losses=[]
        for st in range(0,total,2048):
            ix=order[st:st+2048]; tr=ix[ix<len(fi)]; wa=ix[ix>=len(fi)]-len(fi)
            negatives=sample_pool(pool,wu[wa],rng)
            warm_items=np.c_[wi[wa],negatives];real_items=fit['items'][tr]
            if a.warm_density:
                background=sample_pool(pool,wu[wa],rng)
                warm_items=np.c_[warm_items,background]
                real_items=np.c_[real_items,np.zeros((len(tr),9),dtype=real_items.dtype)]
            c=torch.tensor(np.concatenate([real_items,warm_items]))
            u=torch.tensor(np.r_[fit['user'][tr],wu[wa]])
            context=torch.tensor(np.r_[fit['context'][tr],wx[wa]])
            rng_state=torch.get_rng_state() if a.antithetic else None
            z=net(c,u,True,context);draws=[z];nt=len(tr)
            if a.antithetic:
                torch.set_rng_state(rng_state);draws.append(net(c,u,-1,context))
                for _ in range(a.latent_pairs-1):
                    rng_state=torch.get_rng_state();draws.append(net(c,u,True,context))
                    torch.set_rng_state(rng_state);draws.append(net(c,u,-1,context))
            real=sum(nn.functional.cross_entropy(torch.cat([zz[:nt,:10],torch.zeros(nt,1)],1),torch.tensor(fit['target'][tr]),reduction='sum') for zz in draws)/len(draws)
            if a.marginal_real:
                log_probability=torch.stack([torch.log_softmax(torch.cat([zz[:nt,:10],torch.zeros(nt,1)],1),1) for zz in draws])
                log_probability=torch.logsumexp(log_probability,0)-math.log(len(draws))
                real=nn.functional.nll_loss(log_probability,torch.tensor(fit['target'][tr]),reduction='sum')
            warm=sum(nn.functional.cross_entropy(zz[nt:,:10],torch.zeros(len(wa),dtype=torch.long),reduction='sum') for zz in draws)/len(draws)
            if a.warm_density:
                density=0
                for zz in draws:
                    log_background=torch.logsumexp(torch.cat([zz[nt:,10:],torch.zeros(len(wa),1)],1),1)
                    log_density=nn.functional.logsigmoid(zz[nt:,:10]-log_background[:,None])
                    density=density+nn.functional.cross_entropy(log_density,torch.zeros(len(wa),dtype=torch.long),reduction='sum')/len(draws)
                warm=(1-a.warm_density)*warm+a.warm_density*density
            mu=net.exposure_user.weight; ls=net.exposure_logstd.weight.clamp(-4,1)
            extra=.5*((mu-prior_mean).square().sum()/a.variance+(mu.shape[-1]//ls.shape[-1])*((2*ls).exp()/a.variance-1+math.log(a.variance)-2*ls).sum())
            extra=extra+10*net.item_bias.weight.square().sum()
            if a.exposure_item_penalty>0:extra=extra+a.exposure_item_penalty*(net.exposure_items-exposure_item_anchor).square().sum()
            loss=(real+a.warm_weight*warm)/len(ix)+(a.preference_kl*net.kl()+.5*extra)/total
            if a.rank_weight:
                probability=torch.stack([torch.softmax(torch.cat([zz[:nt,:10],torch.zeros(nt,1)],1),1)[:,:10] for zz in draws]).mean(0).ravel().clamp(1e-6,1-1e-6)
                logits=torch.logit(probability);labels=(torch.arange(10)[None,:]==torch.tensor(fit['target'][tr])[:,None]).ravel()
                positive=logits[labels];negative=logits[~labels]
                if len(positive) and len(negative):
                    sampled=negative[torch.randint(len(negative),(len(positive),32))]
                    loss=loss+a.rank_weight*(nt/len(ix))*nn.functional.softplus(sampled-positive[:,None]).mean()
            opt.zero_grad(set_to_none=True);loss.backward();opt.step();losses.append(loss.item())
        rec={'epoch':epoch,'seconds':time.time()-start,'loss':float(np.mean(losses))}
        records.append(rec);print(rec,flush=True)
    torch.save({'state':net.state_dict(),'cut':a.cut,'epoch':a.epochs},out/'residual.pt')
    np.savez_compressed(out/'prior.npz',mean=prior_mean.numpy(),ridge_coef=ridge.coef_,ridge_intercept=ridge.intercept_)
    (out/'training.json').write_text(json.dumps({'args':vars(a),'prior':prior_report,'records':records},indent=2))


if __name__=='__main__': main()
