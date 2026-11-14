"""Regenerate V5 from the packaged models and compact evaluation inputs."""
import argparse,json,gc
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import lightgbm as lgb
from infer_base_v5 import predict as base_predict
from variational import Variational,predict as variational_predict
from cluster_unit import Latent as UnitLatent,pred as unit_predict
from stack_v5 import additional
from exposure_features import state as exposure_state,features as exposure_features
from common import subset
from validate_submission import validate

def predict(root,threads=4):
    root=Path(root);models=root/'models';data=root/'data'
    ids,base,x,names=base_predict(root,threads,True)
    d=dict(np.load(data/'evaluation.npz'))
    old=pd.read_csv(data/'v2_submission.csv',dtype={'rowid':str})
    assert np.array_equal(ids,old.rowid.to_numpy())
    pc=old.predicted_probability.to_numpy().reshape(-1,10).sum(1,keepdims=True)
    ck=torch.load(models/'variational_preference.pt',map_location='cpu',weights_only=True)
    s=ck['state'];v=s['initial'].numpy();u=s['user.weight'].numpy()
    net=Variational(v,u,np.zeros(len(v)),np.zeros(len(u)),np.zeros(len(u),np.float32))
    net.load_state_dict(s);z,h=variational_predict(net,d);del net
    us=torch.load(models/'cluster_unit_preference.pt',map_location='cpu',weights_only=True)['state']
    unet=UnitLatent(us['initial'].numpy(),us['user.weight'].numpy());unet.load_state_dict(us)
    uu,hh=unit_predict(unet,d,np.ones((len(d['user']),1)));del unet,us
    extra,en=additional({'user':z,'history':h},s,d,{'user':uu,'history':hh},pc)
    del s,ck,z,h,uu,hh;gc.collect()
    train=dict(np.load(data/'train.npz'));audit=json.loads((data/'audit.json').read_text())
    exp=exposure_state(train,audit['n_items'],audit['n_users'],1737936000,data/'exposure_basis.npz')
    del train;gc.collect();parts=[]
    for start in range(0,len(d['user']),4096):
        block,exn=exposure_features(subset(d,np.arange(start,min(start+4096,len(d['user'])))),exp);parts.append(block)
    ex=np.concatenate(parts);xx=np.c_[x,extra,ex];fnames=names+en+exn
    model=lgb.Booster(model_file=str(models/'v5_calibration.txt'))
    assert model.feature_name()==fnames
    p=model.predict(xx,num_threads=threads)
    spec=json.loads((models/'v5_spec.json').read_text());w=spec['v5_weight']
    return ids,np.clip(w*p+(1-w)*base,1e-9,1-1e-9),xx,fnames
