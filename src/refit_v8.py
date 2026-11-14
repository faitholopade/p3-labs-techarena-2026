"""Refit the selected uncertainty-aware AUC calibration heads."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from calibration_model import fit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v4',required=True);ap.add_argument('--v5',required=True);ap.add_argument('--v7',required=True);ap.add_argument('--posterior',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    b=np.load(Path(a.v4)/'validation/v4/meta_training.npz');e=np.load(Path(a.v5)/'validation/v5/added_features.npz');r=np.load(Path(a.v7)/'validation/v7/residual_features.npz');p=np.load(a.posterior)
    for key in ['indices','labels','fold']:assert np.array_equal(b[key],e[key])
    assert np.array_equal(b['indices'],r['indices']) and np.array_equal(b['indices'],p['indices']);x=np.c_[b['features'],e['features'],r['features'],p['features']];y=b['labels'];names=json.loads((Path(a.v5)/'validation/v5/features.json').read_text())+r['names'].tolist()+p['names'].tolist()
    for seed in [731,932,1417]:
        ck=fit(x,y,names,seed=seed,epochs=20,swa_start=12,width=32,selection='compact',architecture='plain',rank_weight=1.);torch.save(ck,out/f'v8_calibration_{seed}.pt');print('Saved seed',seed,flush=True)
    (out/'spec.json').write_text(json.dumps({'seeds':[731,932,1417],'epochs':20,'swa_start':12,'width':32,'rank_weight':1.,'new_weight':.75,'rows':len(y),'input_features':x.shape[1],'selected_features':len(ck['selected'])},indent=2))

if __name__=='__main__':main()
