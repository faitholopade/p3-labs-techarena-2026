"""Reproduce the Phase 1 V3 submission from the packaged models and inputs."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from scipy.special import expit
from features import block
from latent import Latent,pred
from stack import meta_features
from validate_submission import validate

def save_state(path,state):
    s,counts,ucount,profiles=state;arrays={f's_{k}':v for k,v in s.items()};arrays.update(counts=counts,ucount=ucount)
    for j,profile in enumerate(profiles):
        for k,arr in zip(['v','norm','users','neg'],profile):arrays[f'p{j}_{k}']=arr
    np.savez_compressed(path,**arrays)

def load_state(path):
    a=dict(np.load(path));s={k[2:]:v for k,v in a.items() if k.startswith('s_')};s['g']=float(s['g']);profiles=[tuple(a[f'p{j}_{k}'] for k in ['v','norm','users','neg']) for j in range(3)]
    return s,a['counts'],a['ucount'],profiles

def combine(old,tree,user,history,x,names,modeldir,threads=4):
    xm,meta_names=meta_features(old,tree,user,history,x,names);linear=np.load(modeldir/'calibration_linear.npz');pl=expit(((xm[:,:5]-linear['mean'])/linear['scale'])@linear['coef']+linear['intercept'])
    model=lgb.Booster(model_file=str(modeldir/'calibration_tree.txt'));assert model.feature_name()==meta_names;pt=model.predict(xm,num_threads=threads)
    return np.clip(.5*pl+.5*pt,1e-9,1-1e-9)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='P3_Labs_submission_v3.csv');ap.add_argument('--threads',type=int,default=4);a=ap.parse_args();root=Path(a.root);models=root/'models';data=root/'data';torch.set_num_threads(a.threads)
    d=dict(np.load(data/'evaluation.npz'));ids=np.load(data/'evaluation_rowids.npy');old=pd.read_csv(data/'v2_submission.csv',dtype={'rowid':str});assert np.array_equal(old.rowid.to_numpy(),ids);old=old.predicted_probability.to_numpy().reshape(-1,10)
    x,names=block(d,np.arange(len(d['user'])),load_state(models/'feature_state.npz'));tree=lgb.Booster(model_file=str(models/'preference_tree.txt'));assert tree.feature_name()==names;p_tree=tree.predict(x,num_threads=a.threads).reshape(-1,10);print('Tree predictions complete',flush=True)
    ck=torch.load(models/'preference_neural.pt',map_location='cpu',weights_only=True);state=ck['state'];network=Latent(state['initial'].numpy(),state['user.weight'].numpy());network.load_state_dict(state);u,h=pred(network,d,np.ones((len(d['user']),1)));print('Neural predictions complete',flush=True)
    p=combine(old,p_tree,u,h,x,names,models,a.threads);pd.DataFrame({'rowid':ids,'predicted_probability':p}).to_csv(a.out,index=False,float_format='%.10g');print(json.dumps(validate(a.out,ids),indent=2))

if __name__=='__main__':main()
