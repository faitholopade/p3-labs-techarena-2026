"""Condition exposure coefficients on the observed candidate slate, without labels.

The supplied prior is fitted only to pre-cutoff impression profiles and factors.
Each query user must appear once. No query outcome or future event is accessed.
"""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from common import load_data,subset


def condition(state,fit,query,coefficient,variance=.5,strength=1.,mode='posterior'):
    users=np.asarray(query['user']);items=np.asarray(query['items'])
    if len(np.unique(users))!=len(users):raise ValueError('Expected one observed query slate per user.')
    q=state['exposure_items'].numpy();nu=len(state['exposure_user.weight'])
    count=np.bincount(fit['user'],minlength=nu);profile=np.zeros((nu,q.shape[1]),np.float64)
    for start in range(0,len(fit['user']),4096):
        sl=slice(start,start+4096);np.add.at(profile,fit['user'][sl],q[fit['items'][sl]].sum(1))
    profile/=np.maximum(10*count[:,None],1)
    delta_profile=(q[items].mean(1)-profile[users])/(count[users,None]+1)
    delta_prior=delta_profile@coefficient.T
    uncertainty=state['exposure_logstd.weight'].numpy()[users]
    multiplier=np.minimum(np.exp(2*np.clip(uncertainty,-4,1))/variance,1) if mode=='posterior' else np.ones_like(uncertainty)
    adjustment=strength*multiplier*delta_prior
    result={k:v.clone() for k,v in state.items()}
    result['exposure_user.weight'][users]+=torch.tensor(adjustment,dtype=state['exposure_user.weight'].dtype)
    report={'query_users':len(users),'mode':mode,'strength':strength,'mean_adjustment_norm':float(np.linalg.norm(adjustment,axis=1).mean()),'max_adjustment_norm':float(np.linalg.norm(adjustment,axis=1).max()),'mean_posterior_multiplier':float(multiplier.mean()),'query_columns_used':['user','items'],'fit_uses_labels':False}
    return result,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','factor','prior','query','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--variance',type=float,default=.5)
    p.add_argument('--strength',type=float,default=1.);p.add_argument('--mode',choices=['posterior','direct'],default='posterior')
    a=p.parse_args();d,_,_=load_data(a.root/'data');fit=subset(d,np.flatnonzero(d['timestamp']<a.cut));query=np.load(a.query)
    checkpoint=torch.load(a.factor,map_location='cpu',weights_only=False);assert checkpoint['cut']==a.cut
    prior=np.load(a.prior);state,report=condition(checkpoint['state'],fit,query,prior['ridge_coef'],a.variance,a.strength,a.mode)
    a.out.mkdir(parents=True,exist_ok=True);checkpoint['state']=state;torch.save(checkpoint,a.out/'residual.pt')
    report.update({'cut':a.cut,'source_factor_sha256':hashlib.sha256(a.factor.read_bytes()).hexdigest(),'prior_sha256':hashlib.sha256(a.prior.read_bytes()).hexdigest()})
    (a.out/'conditioning.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':main()
