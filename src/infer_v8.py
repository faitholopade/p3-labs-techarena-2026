"""Regenerate V8 from the packaged compact inputs and saved models."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from infer_v7_components import predict as v7_components
from posterior_inference import features as posterior_features
from calibration_model import predict as calibrated_predict
from validate_submission import validate

def predict(root,threads=3):
    root=Path(root);ids,p7,x,names,d,state=v7_components(root,threads);spec=json.loads((root/'models/v8_spec.json').read_text());extra,en=posterior_features(state,d,spec['qmc_base_samples']);x=np.c_[x,extra];names+=en;pred=[]
    for filename in spec['models']:
        ck=torch.load(root/'models'/filename,map_location='cpu',weights_only=False);assert ck['names']==names;pred.append(calibrated_predict(ck,x))
    w=spec['new_weight'];p=(1-w)*p7+w*np.mean(pred,axis=0);assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1;assert p.reshape(-1,10).sum(1).max()<=1+1e-12;return ids,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='P3_Labs_submission_v8.csv');ap.add_argument('--threads',type=int,default=3);a=ap.parse_args();ids,p=predict(a.root,a.threads);pd.DataFrame({'rowid':ids,'predicted_probability':p}).to_csv(a.out,index=False,float_format='%.10g');print(json.dumps(validate(a.out,ids),indent=2),flush=True)

if __name__=='__main__':main()
