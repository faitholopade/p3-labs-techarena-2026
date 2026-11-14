"""Restore V13's matched validation features and verify frozen-head parity."""
import sys,json,gc,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from common import load_data,subset
from daily_inference import build
from sequence_calibration import history_pairs,predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extension',required=True);p.add_argument('--v12-extension',required=True);p.add_argument('--cache',required=True)
    a=p.parse_args();root=Path(a.root);extension=Path(a.extension);cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True)
    data,_,audit=load_data(root/'data');e=dict(np.load(extension/'validation/v13/evidence.npz'))
    e12=dict(np.load(Path(a.v12_extension)/'validation/v12/evidence.npz'))
    assert np.array_equal(e['indices'],e12['indices']) and np.array_equal(e['labels'],e12['labels'])
    # V12 evidence stores the rank branch already blended 50:50 with V8.
    legacy=(.625*e12['v11']+.625*e12['rank']-e12['v12'])/.25
    assert np.isfinite(legacy).all() and legacy.min()>=0 and legacy.max()<=1
    np.save(cache/'legacy.npy',legacy)
    ix=e['indices'];fold=e['fold'].reshape(-1,10)[:,0];reconstructed=np.empty(len(ix)*10)
    for name,cut in [('earlier',1737763200),('later',1737849600)]:
        select=np.flatnonzero((data['timestamp'][ix]>=cut)&(data['timestamp'][ix]<cut+86400))
        query=subset(data,ix[select]);prior=subset(data,np.flatnonzero(data['timestamp']<cut))
        model=root/'validation/v11/source_models'/f'{name}_joint24.pt'
        if (cache/f'{name}_raw.npy').exists():
            raw=np.load(cache/f'{name}_raw.npy',mmap_mode='r');names=json.loads((cache/f'{name}_names.json').read_text())
        else:
            raw,names=build(prior,query,cut,audit['n_items'],audit['n_users'],model,state_root=extension/'validation/source_states')
            np.save(cache/f'{name}_raw.npy',raw)
        np.savez_compressed(cache/f'{name}_query.npz',**query)
        np.savez_compressed(cache/f'{name}_alignment.npz',selection=select,fold=fold[select])
        (cache/f'{name}_names.json').write_text(json.dumps(names))
        del prior;gc.collect()
        state=torch.load(model,map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,query)
        pred=np.empty(len(query['user'])*10)
        for f in range(3):
            local=np.flatnonzero(fold[select]==f);rows=(local[:,None]*10+np.arange(10)).ravel()
            pieces=[]
            for seed in [731,932,1417]:
                ck=torch.load(extension/'validation/v13/fold_models'/f'head_{f}_{seed}.pt',map_location='cpu',weights_only=False)
                assert ck['base_spec']['names']==names
                pieces.append(predict(ck,raw[rows],pairs[local],mask[local]))
            pred[rows]=np.mean(pieces,axis=0)
        rows=(select[:,None]*10+np.arange(10)).ravel();reconstructed[rows]=pred
        error=float(np.max(np.abs(.375*legacy[rows]+.625*pred-e['sequence'][rows])));print({'period':name,'events':len(query['user']),'parity_max_abs':error},flush=True)
        assert error<1e-6
        del raw,pairs,query;gc.collect()
    np.save(cache/'original_sequence.npy',reconstructed)
    (cache/'parity.json').write_text(json.dumps({'sequence_candidate_max_abs':float(np.max(np.abs(.375*legacy+.625*reconstructed-e['sequence']))),'rows':len(reconstructed)},indent=2))


if __name__=='__main__':main()
