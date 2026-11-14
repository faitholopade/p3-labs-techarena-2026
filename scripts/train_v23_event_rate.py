"""Train a fixed chronological propensity correction on all pre-cutoff slates."""
import argparse,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,lightgbm as lgb
from common import load_data,subset
from history_choice import item_state
from chronological_event_rate import build,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','factor','query','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.time()
    data,_,audit=load_data(a.root/'data');fit=subset(data,np.flatnonzero(data['timestamp']<a.cut));del data
    ck=torch.load(a.factor,map_location='cpu',weights_only=False);assert int(ck['cut'])==a.cut;items=item_state(ck['state']);del ck
    query=dict(np.load(a.query));assert np.all(query['target']==-1)
    x,offset,names=build(fit,query,items,audit['n_users'],a.out/'features.npy');n=len(fit['user'])
    print({'prepared_events':n,'columns':len(names),'seconds':time.time()-start},flush=True)
    params={'objective':'binary','metric':'None','learning_rate':.025,'num_leaves':7,'min_data_in_leaf':1000,'lambda_l2':100.,'max_bin':127,'feature_fraction':.85,'bagging_fraction':.85,'bagging_freq':1,'num_threads':a.threads,'seed':20261112,'verbosity':-1,'force_col_wise':True,'deterministic':True}
    model=lgb.train(params,lgb.Dataset(x[:n],label=(fit['target']<10),init_score=offset[:n],free_raw_data=True),num_boost_round=400);model.save_model(str(a.out/'model.txt'))
    event=predict(model,x[n:],offset[n:],a.threads);assert np.isfinite(event).all() and np.all((event>=0)&(event<=1));np.savez_compressed(a.out/'prediction.npz',event_rate=event)
    with a.factor.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    spec={'cut':a.cut,'factor_file':str(a.factor),'factor_sha256':sha,'training_events':n,'query_events':len(query['user']),'max_training_timestamp':int(fit['timestamp'].max()),'names':names,'params':params,'rounds':400,'offset':'logit(activity_user_rate10)','query_targets_masked':True,'activity_uses_strictly_prior_timestamps':True,'seconds':time.time()-start,'limits':'Weekend-only confirmation; item representations trained on the same pre-cutoff window. No public-score prediction.'}
    (a.out/'spec.json').write_text(json.dumps(spec,indent=2)+'\n');print({'complete':str(a.out),'seconds':time.time()-start},flush=True)


if __name__=='__main__':main()
