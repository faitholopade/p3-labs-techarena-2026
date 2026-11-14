"""Reproduce the V9 submission from saved models and compact contest data."""
import argparse,json,gc
from pathlib import Path
import numpy as np,pandas as pd,torch
from infer_v8 import predict as v8_predict
from daily_inference import build
from calibration_model import predict as head_predict
from validate_submission import validate

def predict(root,threads=3):
    root=Path(root);ids,old=v8_predict(root,threads);gc.collect();train=dict(np.load(root/'data/train.npz'));ev=dict(np.load(root/'data/evaluation.npz'));audit=json.loads((root/'data/audit.json').read_text());x,names=build(train,ev,1737936000,audit['n_items'],audit['n_users'],root/'models/exposure_residual.pt',feature_state=root/'models/feature_state.npz',exposure_basis=root/'data/exposure_basis.npz');spec=json.loads((root/'models/v9_spec.json').read_text());pred=[]
    for filename in spec['models']:
        ck=torch.load(root/'models'/filename,map_location='cpu',weights_only=False);assert ck['names']==names;pred.append(head_predict(ck,x))
    p=(1-spec['new_weight'])*old+spec['new_weight']*np.mean(pred,axis=0);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1;assert p.reshape(-1,10).sum(1).max()<=1+1e-12;return ids,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='P3_Labs_submission_v9.csv');ap.add_argument('--threads',type=int,default=3);a=ap.parse_args();ids,p=predict(a.root,a.threads);pd.DataFrame({'rowid':ids,'predicted_probability':p}).to_csv(a.out,index=False,float_format='%.10g');print(json.dumps(validate(a.out,ids),indent=2),flush=True)
if __name__=='__main__':main()
