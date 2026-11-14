"""Refit V7's three smooth heads from the recorded chronological calibration data."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from neural_head import fit

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--v4',required=True);ap.add_argument('--v5',required=True);ap.add_argument('--extra',required=True);ap.add_argument('--out',required=True);a=ap.parse_args();v4=Path(a.v4);v5=Path(a.v5);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    b=np.load(v4/'validation/v4/meta_training.npz');e=np.load(v5/'validation/v5/added_features.npz');r=np.load(a.extra)
    for key in ['indices','labels','fold']:assert np.array_equal(b[key],e[key])
    assert np.array_equal(b['indices'],r['indices']);x=np.c_[b['features'],e['features'],r['features']];y=b['labels'];names=json.loads((v5/'validation/v5/features.json').read_text())+r['names'].tolist()
    for seed in [731,932,1417]:
        ck=fit(x,y,names,seed=seed,epochs=20,swa_start=12,width=32);torch.save(ck,out/f'neural_calibration_{seed}.pt');print('Saved seed',seed,flush=True)
    (out/'spec.json').write_text(json.dumps({'seeds':[731,932,1417],'epochs':20,'swa_start':12,'width':32,'neural_weight':.75,'rows':len(y),'input_features':x.shape[1],'selected_features':len(ck['selected'])},indent=2))

if __name__=='__main__':main()
