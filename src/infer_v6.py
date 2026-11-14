"""Regenerate the selected V6 CSV from packaged models and compact inputs."""
import argparse,json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from infer_v5_components import predict as v5_components
from event_choice import probabilities,combine
from validate_submission import validate

def predict(root,threads=4):
    root=Path(root);ids,p5,x,names=v5_components(root,threads)
    model=lgb.Booster(model_file=str(root/'models/event_choice_calibration.txt'))
    assert model.feature_name()==names
    group=probabilities(model.predict(x,num_threads=threads));spec=json.loads((root/'models/v6_spec.json').read_text())
    p=combine(p5,group,spec['event_choice_weight'])
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    assert np.max(p.reshape(-1,10).sum(1))<=1+1e-12
    return ids,p

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='P3_Labs_submission_v6.csv');ap.add_argument('--threads',type=int,default=4);a=ap.parse_args()
    ids,p=predict(a.root,a.threads);pd.DataFrame({'rowid':ids,'predicted_probability':p}).to_csv(a.out,index=False,float_format='%.10g');print(json.dumps(validate(a.out,ids),indent=2),flush=True)

if __name__=='__main__':main()
