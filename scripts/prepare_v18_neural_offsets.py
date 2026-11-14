"""Use only heads excluding each outer validation user to initialise its tree.

Meta-training predictions use those same frozen outer-fold heads. They can be
in-sample for meta-training users; no outer validation labels enter either head.
"""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sequence_calibration import history_pairs,predict


def utility(probability):
    p=probability.reshape(-1,10).astype(np.float64)
    return np.log(np.maximum(p,1e-9)/np.maximum(1-p.sum(1,keepdims=True),1e-9)).ravel()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','cache','v13-extension','v17-extension','v17-experiments','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);raw=np.load(a.features/'raw.npy',mmap_mode='r');offsets=np.empty((4,len(raw)),np.float64);errors={}
    for period in ['earlier','later']:
        query=dict(np.load(a.cache/f'{period}_query.npz'));align=np.load(a.cache/f'{period}_alignment.npz');sel=align['selection'];rows=(sel[:,None]*10+np.arange(10)).ravel();x=raw[rows]
        state=torch.load(a.v17_extension/f'validation/factors/{period}_factor_users_only.pt',map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,query)
        parent=np.load(a.v17_experiments/f'{period}_users_only/screen256/predictions.npz');assert np.array_equal(parent['selection'],sel)
        for f in range(4):
            paths=[a.v13_extension/(f'validation/v13/fold_models/head_{f}_{seed}.pt' if f<3 else f'models/sequence_{seed}.pt') for seed in [731,932,1417]]
            preds=[predict(torch.load(path,map_location='cpu',weights_only=False),x,pairs,mask) for path in paths];prob=np.mean(preds,axis=0);offsets[f,rows]=utility(prob)
            if f<3:
                held=np.repeat(align['fold']==f,10);error=float(np.max(np.abs(prob[held]-parent['sequence'][held])));assert error<1e-6;errors[f'{period}_{f}']=error
            print({'period':period,'fold':f if f<3 else 'full'},flush=True)
        del x,pairs,state;gc.collect()
    np.savez_compressed(a.out/'offsets.npz',fold0=offsets[0],fold1=offsets[1],fold2=offsets[2],full=offsets[3])
    (a.out/'report.json').write_text(json.dumps({'heldout_sequence_parity':errors,'outer_user_exclusion_preserved':True,'meta_training_predictions':'Same outer-fold neural heads; in-sample for some training users, requiring independent outer evaluation.'},indent=2))


if __name__=='__main__':main()
