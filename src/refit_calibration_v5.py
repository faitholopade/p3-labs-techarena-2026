"""Refit V5 calibration from the packaged held-out features."""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--v4-root',required=True,help='Directory of the previously supplied V4 archive');ap.add_argument('--out',default='refit_v5_calibration.txt');a=ap.parse_args();root=Path(a.root)
z=np.load(root/'validation/v5/added_features.npz');base=np.load(Path(a.v4_root)/'validation/v4/meta_training.npz')
for key in ['labels','indices','fold']:assert np.array_equal(z[key],base[key]),'V4/V5 calibration rows must align'
x=np.c_[base['features'],z['features']];spec=json.loads((root/'validation/v5/metrics.json').read_text());names=json.loads((root/'validation/v5/features.json').read_text())
assert x.shape[1]==len(names)
model=lgb.train(spec['params'],lgb.Dataset(x,label=z['labels'],feature_name=names),num_boost_round=600);model.save_model(a.out);print('Saved',a.out)
