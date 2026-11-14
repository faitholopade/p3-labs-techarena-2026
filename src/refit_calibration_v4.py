"""Refit the selected 600-round calibration tree from supplied held-out features."""
import argparse,json
from pathlib import Path
import numpy as np
import lightgbm as lgb
ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--out',default='refit_flex_calibration.txt');a=ap.parse_args();root=Path(a.root);z=np.load(root/'validation/v4/meta_training.npz');spec=json.loads((root/'validation/v4/metrics.json').read_text());names=json.loads((root/'validation/v4/features.json').read_text());model=lgb.train(spec['params'],lgb.Dataset(z['features'],label=z['labels'],feature_name=names),num_boost_round=600);model.save_model(a.out);print('Saved',a.out)
