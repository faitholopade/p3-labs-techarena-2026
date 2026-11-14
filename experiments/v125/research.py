"""Conditional directional importance correction, with causal input boundaries."""
import argparse, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import log_ndtr, pbdv, softmax
from sklearn.metrics import roc_auc_score, log_loss

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'runtime/v125'

def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()

def write(p, obj):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(obj, indent=2)+'\n')

def log_radial_integral(a):
    """log integral_0^infty r^7 exp(-r*r/2+a*r) dr, for 8D directions.

    The radial substitution r=radius/sigma leaves only this direction-dependent
    term in the projected-normal density. Constants cancel within each user.
    """
    a = np.asarray(a, np.float64)
    out = np.empty_like(a); pos = a >= 0; x = a[pos]
    i0 = .5*x*x + .5*np.log(2*np.pi) + log_ndtr(x)
    la = np.full_like(x, -np.inf); np.log(x, out=la, where=x>0)
    i1 = np.logaddexp(0, la+i0)
    for k in range(2, 8): i0, i1 = i1, np.logaddexp(la+i1, np.log(k-1)+i0)
    out[pos] = i1
    x = a[~pos]
    out[~pos] = math.lgamma(8) + x*x/4 + np.log(pbdv(-8, -x)[0])
    assert np.isfinite(out).all()
    return out

def noise():
    engine = torch.quasirandom.SobolEngine(96, scramble=True, seed=920)
    z = torch.erfinv(2*engine.draw(128).clamp(1e-6, 1-1e-6)-1)*math.sqrt(2)
    return torch.cat([z, -z], 0)

def directions(state, eps):
    return torch.nn.functional.normalize(state['user.weight'][:,None] +
        state['user_logstd.weight'].clamp(-3,1).exp()[:,None]*eps[None,:,:8], dim=-1)

def warmup(fit):
    mask = (fit['hist']>0) & (fit['hist_ts']<1737331200)
    u = np.broadcast_to(fit['user'][:,None], mask.shape)[mask]
    i, t = fit['hist'][mask], fit['hist_ts'][mask]
    z = np.rec.fromarrays([u,i,t], names='u,i,t')
    _, index = np.unique(z, return_index=True)
    return u[index], i[index]

def sample_pool(pool, users, rng, n):
    group = pool['labels'][users]; ptr = pool['ptr']
    loc = ptr[group,None] + (rng.random((len(users),n))*(ptr[group+1]-ptr[group])[:,None]).astype(np.int64)
    return pool['pool'][loc]

@torch.no_grad()
def build():
    started = time.time(); plan = json.loads((HERE/'plan.json').read_text())
    factor = ROOT/'runtime/benchmark/inputs/factors/friday.pt'
    assert sha(factor) == plan['factor_sha256']
    ck = torch.load(factor, map_location='cpu', weights_only=False)
    assert ck['cut'] == plan['cutoff']; s=ck['state']
    data = np.load(ROOT/'runtime/v11/data/train.npz'); ix = np.flatnonzero(data['timestamp']<ck['cut'])
    fit = {k:data[k][ix] for k in ['items','user','timestamp','hist','hist_ts','target']}
    assert fit['timestamp'].max()<ck['cut']
    pool = dict(np.load(ROOT/'runtime/v124/pools/exposure_pools_1737676800_64.npz'))
    eps=noise(); uv=directions(s,eps); nuser=len(uv); particles=len(eps)
    im=s['item.weight']; sd=s['item_logstd.weight'].clamp(-3,1).exp()
    iv=im/(im.square().sum(-1,keepdim=True)+7*sd.square()).sqrt().clamp_min(1e-8)
    beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp()
    eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp()/math.sqrt(8)
    likelihoods=[]; rng=np.random.default_rng(plan['warm_negative_seed']); wu,wi=warmup(fit)
    for phase in ['real','warm']:
        result=torch.zeros(nuser,particles,dtype=torch.float64)
        count=len(fit['user']) if phase=='real' else len(wu); phase_start=time.time()
        for st in range(0,count,1024):
            sl=slice(st,st+1024)
            if phase=='real':
                u=torch.tensor(fit['user'][sl],dtype=torch.long); c=torch.tensor(fit['items'][sl],dtype=torch.long)
            else:
                u=torch.tensor(wu[sl],dtype=torch.long)
                c=torch.tensor(np.c_[wi[sl], sample_pool(pool,wu[sl],rng,plan['warm_negatives'])],dtype=torch.long)
            offset=(s['exposure_items'][c]*s['exposure_user.weight'][u,None]).sum(-1)*eb+s['item_bias.weight'][c,0]+s['outside.weight'][u]
            z=torch.bmm(uv[u],iv[c].transpose(1,2))*beta+offset[:,None]
            if phase=='real':
                z=torch.cat([z,torch.zeros(len(u),particles,1)],2)
                y=torch.tensor(fit['target'][sl],dtype=torch.long)
                ll=z.gather(2,y[:,None,None].expand(-1,particles,1)).squeeze(2)-torch.logsumexp(z,2)
            else:
                ll=z[:,:,0]-torch.logsumexp(z[:,:,1:],2)+math.log(plan['warm_negatives'])
            result.index_add_(0,u,ll.double())
            if st and st%131072==0: print(dict(phase=phase,processed=st,total=count,seconds=time.time()-phase_start),flush=True)
        likelihoods.append(result.numpy());print(dict(phase=phase,events=count,seconds=time.time()-phase_start),flush=True)
    mu=s['user.weight'].numpy(); us=s['user_logstd.weight'].clamp(-3,1).exp().numpy()
    angle=np.einsum('nd,nsd->ns',mu/us,uv.numpy()); lq=log_radial_integral(angle)
    weights={}; reports=[]
    for warm_weight in plan['warm_likelihood_weights']:
        logw=likelihoods[0]+warm_weight*likelihoods[1]-lq
        raw=softmax(logw,axis=1); sumsq=(raw*raw).sum(1)
        alpha=np.minimum(1,np.sqrt((particles/plan['minimum_effective_particles']-1)/np.maximum(particles*sumsq-1,1e-14)))
        w=(1-alpha[:,None])/particles+alpha[:,None]*raw
        ess=1/(w*w).sum(1);assert ess.min()>=plan['minimum_effective_particles']-1e-7
        assert np.allclose(w.sum(1),1,atol=1e-12)
        weights[f'w{int(warm_weight)}']=w.astype(np.float32)
        reports.append(dict(warm_weight=warm_weight,raw_ess_quantiles=np.quantile(1/sumsq,[0,.1,.5,.9,1]).tolist(),mixture_alpha_quantiles=np.quantile(alpha,[0,.1,.5,.9,1]).tolist(),final_ess_min=float(ess.min())))
    np.savez_compressed(OUT/'weights.npz',**weights)
    np.savez_compressed(OUT/'likelihoods.npz',real=likelihoods[0].astype(np.float32),warm=likelihoods[1].astype(np.float32),log_proposal=lq.astype(np.float32))
    write(OUT/'fit.json',dict(plan_sha256=sha(HERE/'plan.json'),source_sha256=sha(Path(__file__)),factor_sha256=sha(factor),pool_sha256=sha(ROOT/'runtime/v124/pools/exposure_pools_1737676800_64.npz'),training_events=len(ix),warm_clicks=len(wu),maximum_training_timestamp=int(fit['timestamp'].max()),query_outcomes_used=False,weights_sha256=sha(OUT/'weights.npz'),weights=reports,seconds=time.time()-started))
    print(dict(stage='fit_complete',seconds=time.time()-started,weights=reports),flush=True)

@torch.no_grad()
def predict():
    s=torch.load(ROOT/'runtime/benchmark/inputs/factors/friday.pt',map_location='cpu',weights_only=False)['state']
    e=np.load(ROOT/'runtime/v120/alignment.npz');data=np.load(ROOT/'runtime/v11/data/train.npz');idx=e['indices']
    items=data['items'][idx];users=data['user'][idx];weights=dict(np.load(OUT/'weights.npz'));names=['original']+list(weights)
    eps=noise();parts={k:[] for k in names};beta=s['log_beta'].clamp(math.log(2),math.log(8)).exp();eb=s['log_exposure_beta'].clamp(math.log(.5),math.log(8)).exp();started=time.time()
    for st in range(0,len(idx),256):
        sl=slice(st,st+256);c=torch.tensor(items[sl],dtype=torch.long);u=torch.tensor(users[sl],dtype=torch.long)
        iv=torch.nn.functional.normalize(s['item.weight'][c][:,None]+s['item_logstd.weight'][c].clamp(-3,1).exp()[:,None]*eps[None,:,8:88].reshape(1,len(eps),10,8),dim=-1)
        eu=(s['exposure_user.weight'][u,None]+s['exposure_logstd.weight'][u,None].clamp(-4,1).exp()*eps[None,:,88:])/math.sqrt(8)
        uv=torch.nn.functional.normalize(s['user.weight'][u,None]+s['user_logstd.weight'][u,None].clamp(-3,1).exp()*eps[None,:,:8],dim=-1)
        z=(iv*uv[:,:,None]).sum(-1)*beta+torch.einsum('nkd,nsd->nsk',s['exposure_items'][c],eu)*eb+s['item_bias.weight'][c,0][:,None]+s['outside.weight'][u,0][:,None,None]
        fullp=torch.softmax(torch.cat([z,torch.zeros(len(u),len(eps),1)],2),2)
        parts['original'].append(fullp.mean(1)[:,:10].numpy());p=fullp[:,:,:10]
        for name in names[1:]: parts[name].append((p*torch.tensor(weights[name][users[sl]])[:,:,None]).sum(1).numpy())
    values={k:np.concatenate(v).astype(np.float64) for k,v in parts.items()};old=np.load(ROOT/'runtime/v124/predictions_cohort1.npz')
    assert np.array_equal(idx,old['indices']);err=float(abs(values['original']-old['original']).max());assert err<2e-7,err
    for name,p in values.items(): assert np.isfinite(p).all() and p.min()>0 and p.sum(1).max()<=1+1e-6
    np.savez_compressed(OUT/'predictions.npz',indices=idx,**values)
    write(OUT/'prediction_audit.json',dict(query_events=len(idx),original_mc_max_error=err,original_parity_tolerance=2e-7,torch_version=torch.__version__,weights_sha256=sha(OUT/'weights.npz'),seconds=time.time()-started,predictions_sha256=sha(OUT/'predictions.npz')))
    print(dict(stage='prediction_complete',original_mc_max_error=err,seconds=time.time()-started),flush=True)

def screen():
    plan=json.loads((HERE/'plan.json').read_text());e=np.load(ROOT/'runtime/v120/evidence.npz');a=np.load(ROOT/'runtime/v120/alignment.npz');p=np.load(OUT/'predictions.npz');assert np.array_equal(e['indices'],p['indices'])
    y=e['labels'];base=e['baseline'];day=(a['timestamp']-plan['cutoff'])//86400
    masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(day==0,10),delayed=np.repeat(day>0,10),third_day=np.repeat(day==2,10));ba={k:roc_auc_score(y[m],base[m]) for k,m in masks.items()};bl=log_loss(y,base);rows=[]
    l0=np.log(p['original']/p['original'].sum(1,keepdims=True));preds={}
    for w in plan['warm_likelihood_weights']:
        v=p[f'w{int(w)}'];d=np.log(v/v.sum(1,keepdims=True))-l0;d-=d.mean(1,keepdims=True);d=plan['delta_cap']*np.tanh(d/plan['delta_cap'])
        for strength in plan['strengths']:
            b=base.reshape(-1,10);candidate=(softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)).ravel()
            gains={k:float(roc_auc_score(y[m],candidate[m])-ba[k]) for k,m in masks.items()};ll=float(log_loss(y,candidate)-bl)
            eligible=gains['all']>=plan['primary_gates']['minimum_auc_gain'] and gains['first_day']>0 and gains['delayed']>0 and gains['third_day']>=0 and ll<=0
            name=f'w{int(w)}_s{strength}';r=dict(name=name,warm_weight=w,strength=strength,gains=gains,logloss_change=ll,primary_eligible=eligible);rows.append(r);preds[name]=candidate;print(r,flush=True)
    np.savez_compressed(OUT/'screen_predictions.npz',indices=e['indices'],**preds)
    write(OUT/'primary_screen.json',dict(baseline_auc=ba,results=rows,eligible=[r for r in rows if r['primary_eligible']],plan_sha256=sha(HERE/'plan.json'),source_sha256=sha(Path(__file__)),official_score=None,portal_uploads=0,scope='Reused corrected-forecast V65 proxy; no V108 validation available'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['fit','predict','screen'],required=True);args=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3)
    {'fit':build,'predict':predict,'screen':screen}[args.stage]()
