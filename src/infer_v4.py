"""Regenerate the Phase 1 V4 CSV from saved models and compact input data."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from features import block
from latent import Latent,pred
from unit_mf import Latent as UnitLatent,pred as unit_pred
from stack import meta_features
from flex_stack import extend
from infer_v3 import load_state,combine
from activity import rolling as activity_rolling,NAMES as ACTIVITY_NAMES
from repeats import rolling as repeats_rolling,NAMES as REPEAT_NAMES
from validate_submission import validate

def extras(train,evaluation,ni,nu):
    nt=len(train['user']);u=np.r_[train['user'],evaluation['user']];t=np.r_[train['timestamp'],evaluation['timestamp']];c=np.r_[train['context'],evaluation['context']];y=np.r_[train['target'],np.full(len(evaluation['user']),-1)];order=np.lexsort((t,u));all_activity=activity_rolling(u,t,c,y,order,nu);activity=all_activity[nt:].copy();del all_activity,order
    items=np.r_[train['items'],evaluation['items']];key=(u.astype(np.int64)[:,None]*(ni+1)+items).ravel();rt=np.repeat(t,10);rc=np.repeat(c,10);labels=np.r_[(np.arange(10)[None,:]==train['target'][:,None]).astype(np.float32).ravel(),np.full(len(evaluation['user'])*10,-1)];order=np.lexsort((rt,key));all_repeats=repeats_rolling(key,rt,rc,labels,order).reshape(-1,10,8);repeats=all_repeats[nt:].copy();return activity,repeats

def predict(root,threads=4):
    models=root/'models';data=root/'data';torch.set_num_threads(threads);d=dict(np.load(data/'evaluation.npz'));ids=np.load(data/'evaluation_rowids.npy');old=pd.read_csv(data/'v2_submission.csv',dtype={'rowid':str});assert np.array_equal(old.rowid.to_numpy(),ids);old=old.predicted_probability.to_numpy().reshape(-1,10)
    x,names=block(d,np.arange(len(d['user'])),load_state(models/'feature_state.npz'));tree=lgb.Booster(model_file=str(models/'preference_tree.txt'));assert tree.feature_name()==names;pt=tree.predict(x,num_threads=threads).reshape(-1,10)
    state=torch.load(models/'preference_neural.pt',map_location='cpu',weights_only=True)['state'];net=Latent(state['initial'].numpy(),state['user.weight'].numpy());net.load_state_dict(state);u,h=pred(net,d,np.ones((len(d['user']),1)));v3=combine(old,pt,u,h,x,names,models,threads);xm,mnames=meta_features(old,pt,u,h,x,names);print('V3 model components complete',flush=True)
    state=torch.load(models/'unit_preference.pt',map_location='cpu',weights_only=True)['state'];net=UnitLatent(state['initial'].numpy(),state['user.weight'].numpy());net.load_state_dict(state);uu,hh=unit_pred(net,d,np.ones((len(d['user']),1)));del state,net;gc.collect()
    train=dict(np.load(data/'train.npz'));audit=json.loads((data/'audit.json').read_text());activity,repeats=extras(train,d,audit['n_items'],audit['n_users']);del train;gc.collect();xx=extend(xm,x,{'user':uu,'history':hh},old.sum(1,keepdims=True),activity,repeats);fnames=mnames+['tree_'+n for n in names]+['unit_logit','unit_user','unit_history','unit_std']+['activity_'+n for n in ACTIVITY_NAMES]+['repeats_'+n for n in REPEAT_NAMES];model=lgb.Booster(model_file=str(models/'flex_calibration.txt'));assert model.feature_name()==fnames;pflex=model.predict(xx,num_threads=threads);p=np.clip(.75*pflex+.25*v3,1e-9,1-1e-9);return ids,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='P3_Labs_submission_v4.csv');ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();ids,p=predict(Path(a.root),a.threads);pd.DataFrame({'rowid':ids,'predicted_probability':p}).to_csv(a.out,index=False,float_format='%.10g');print(json.dumps(validate(a.out,ids),indent=2))

if __name__=='__main__':main()
