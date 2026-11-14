"""Fit nonlinear exposure priors from pre-cutoff impression statistics only.

Targets are already fitted pre-cutoff user coefficients. Diagnostics measure
coefficient prediction, not CTR performance. Dense users receive cross-fitted
prior means so their own target coefficient is excluded from prior fitting.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error
from common import load_data,subset


def build_features(fit,state,nu,ni):
    q=state['exposure_items'].numpy();count=np.bincount(fit['user'],minlength=nu)
    imp=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=(nu,ni)).tocsr()
    divisor=np.maximum(10*count[:,None],1)
    profile=np.asarray(imp@q)/divisor
    norm2=np.sum(profile**2,axis=1,keepdims=True)
    upper=np.triu_indices(q.shape[1]);outer=q[:,upper[0]]*q[:,upper[1]]
    moments=np.asarray(imp@outer)/divisor
    radial=np.c_[profile,profile*norm2]
    return count,{'linear':profile,'radial':radial,'moments':np.c_[radial,moments],
                  'cubic':np.c_[PolynomialFeatures(3,include_bias=False).fit_transform(profile),moments]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','factor','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--mode',choices=['all','linear','radial','moments','cubic'],default='all')
    p.add_argument('--alpha',type=float,default=100.)
    a=p.parse_args();torch.set_num_threads(2)
    d,_,audit=load_data(a.root/'data');fit=subset(d,np.flatnonzero(d['timestamp']<a.cut));del d
    ck=torch.load(a.factor,map_location='cpu',weights_only=False)
    if ck['cut']!=a.cut:raise ValueError('Factor cutoff mismatch.')
    state=ck['state'];nu,ni=audit['n_users']+1,audit['n_items']+1
    count,features=build_features(fit,state,nu,ni);target=state['exposure_user.weight'].numpy()
    dense=np.flatnonzero(count>=10);train,test=train_test_split(dense,test_size=.2,random_state=1729)
    linear=Ridge(alpha=10).fit(features['linear'][train],target[train]);pred=linear.predict(features['linear'][test])
    report={'cut':a.cut,'dense_users':len(dense),'diagnostic_users':len(test),
            'baseline_r2':float(r2_score(target[test],pred)),'baseline_mse':float(mean_squared_error(target[test],pred)),
            'diagnostic_scope':'Predicts already fitted pre-cutoff coefficients; not a CTR validation score.',
            'validation_outcomes_used':False,'variants':{}}
    with a.factor.open('rb') as stream:factor_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    a.out.mkdir(parents=True,exist_ok=True)
    for mode in (['linear','radial','moments','cubic'] if a.mode=='all' else [a.mode]):
        x=features[mode];model=make_pipeline(StandardScaler(),Ridge(alpha=a.alpha))
        model.fit(x[train],target[train]);pred=model.predict(x[test])
        diagnostic={'r2':float(r2_score(target[test],pred)),'mse':float(mean_squared_error(target[test],pred)),'features':x.shape[1],'alpha':a.alpha}
        model.fit(x[dense],target[dense]);mean=model.predict(x)
        for fold in range(3):
            held=dense[dense%3==fold];fitted=dense[dense%3!=fold]
            assert not np.intersect1d(held,fitted).size
            model.fit(x[fitted],target[fitted]);mean[held]=model.predict(x[held])
        assert mean.shape==target.shape and np.isfinite(mean).all()
        np.savez_compressed(a.out/f'prior_{mode}.npz',mean=mean.astype(np.float32),cut=a.cut,source_factor_sha256=factor_sha)
        report['variants'][mode]=diagnostic
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
