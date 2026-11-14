"""Regenerate the selected V7 submission from saved models and compact inputs."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,torch,lightgbm as lgb
from infer_v5_components import predict as v5_components
from event_choice import probabilities,combine
from residual_model import load as load_residual,features as residual_features
from neural_head import predict as neural_predict
from validate_submission import validate

def predict(root,threads=3):
    root=Path(root);ids,p5,x,names=v5_components(root,threads);group=lgb.Booster(model_file=str(root/'models/event_choice_calibration.txt'));assert group.feature_name()==names;v6=combine(p5,probabilities(group.predict(x,num_threads=threads)))
    spec=json.loads((root/'models/v7_spec.json').read_text())
    if spec['exposure_residual']:
        d=dict(np.load(root/'data/evaluation.npz'));net=load_residual(root/'models/exposure_residual.pt');extra,en=residual_features(net,d);x=np.c_[x,extra];names+=en
    predictions=[]
    for filename in spec['neural_models']:
        ck=torch.load(root/'models'/filename,map_location='cpu',weights_only=False);assert ck['names']==names;predictions.append(neural_predict(ck,x))
    neural=np.mean(predictions,axis=0);w=spec['neural_weight'];p=(1-w)*v6+w*neural
    assert np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    assert p.reshape(-1,10).sum(1).max()<=1+1e-12
    return ids,p,x,names,d,net.state_dict()
