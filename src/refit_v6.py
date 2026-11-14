"""Refit V6's added calibration using the previously supplied V4/V5 features."""
from pathlib import Path
import argparse,json,numpy as np,lightgbm as lgb
from event_choice import objective
ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--v4-root',required=True);ap.add_argument('--v5-root',required=True);ap.add_argument('--out',default='refit_event_choice_calibration.txt');a=ap.parse_args()
root=Path(a.root);base=np.load(Path(a.v4_root)/'validation/v4/meta_training.npz');extra=np.load(Path(a.v5_root)/'validation/v5/added_features.npz')
for k in ['indices','labels','fold']:assert np.array_equal(base[k],extra[k])
x=np.c_[base['features'],extra['features']];names=json.loads((root/'validation/v6/features.json').read_text());spec=json.loads((root/'validation/v6/spec.json').read_text());params=spec['params'];params['objective']=objective;assert x.shape[1]==len(names)
model=lgb.train(params,lgb.Dataset(x,label=base['labels'],feature_name=names),num_boost_round=spec['rounds']);model.save_model(a.out);print('Saved',a.out)
