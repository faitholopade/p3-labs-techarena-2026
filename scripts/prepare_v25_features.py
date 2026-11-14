"""Restore expanded V20 features and add a frozen pre-day exposure memory."""
import argparse, gc, hashlib, json, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'src'))
import numpy as np
import torch
from common import load_data, subset, binary_labels
from daily_inference import build
from history_tree_features import extend
from v25_exposure_memory import features as memory_features
from calibration_model import predict

def digest(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','v13','v17','v20-features','v20-neural','out']:
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args(); a.out.mkdir(parents=True,exist_ok=True); torch.set_num_threads(3)
    plan={'cutoffs':[1737763200,1737849600], 'posterior_samples':256,
          'query_labels_masked':True,'snapshot':'All exposure outcomes strictly before UTC day cutoff.',
          'memory_windows':[1,3,5], 'base_columns':344,
          'source_sha256':{name:digest(REPO/name) for name in ['src/v25_exposure_memory.py','scripts/prepare_v25_features.py']},
          'data_sha256':digest(a.root/'data/train.npz')}
    path=a.out/'plan.json'
    if path.exists() or (a.out/'raw.npy').exists():raise ValueError('Use a fresh output directory.')
    path.write_text(json.dumps(plan,indent=2)+'\n')
    data,_,audit=load_data(a.root/'data')
    alignment=dict(np.load(a.v20_features/'alignment.npz'))
    indices=alignment['indices']; position=alignment['matched_position']
    assert np.array_equal(binary_labels(subset(data,indices)).ravel(),alignment['labels'])
    assert np.array_equal(data['user'][indices],alignment['user'])
    old_names=json.loads((a.v20_features/'names.json').read_text())
    dest=None; checks=[]; start=time.time()
    for period,cut in zip(['earlier','later'],plan['cutoffs']):
        local=np.flatnonzero(alignment['period']==(['earlier','later'].index(period)))
        ix=indices[local]; query=subset(data,ix);query['target']=np.full(len(ix),-1)
        assert data['timestamp'][ix].min()>=cut and data['timestamp'][ix].max()<cut+86400
        fit=subset(data,np.flatnonzero(data['timestamp']<cut))
        factor=a.v17/f'validation/factors/{period}_factor_users_only.pt'
        ck=torch.load(factor,map_location='cpu',weights_only=False);assert ck['cut']==cut
        print({'stage':'base_features','period':period,'events':len(ix)},flush=True)
        raw,names=build(fit,query,cut,audit['n_items'],audit['n_users'],factor,
                        state_root=a.v13/'validation/source_states',posterior_samples=256)
        memory,mnames=memory_features(fit,query,ck['state'],cut)
        changed=subset(query,np.arange(min(64,len(ix))));changed['target']=np.arange(len(changed['user']))%11
        check_mem,_=memory_features(fit,changed,ck['state'],cut)
        assert np.array_equal(check_mem,memory[:len(changed['user'])*10])
        if dest is None:
            all_names=old_names+mnames
            dest=np.lib.format.open_memmap(a.out/'raw.npy',mode='w+',dtype=np.float32,shape=(len(indices)*10,len(all_names)))
        for begin in range(0,len(ix),4000):
            end=min(begin+4000,len(ix));q=subset(query,np.arange(begin,end))
            extended,en=extend(raw[begin*10:end*10],q,ck['state'],names)
            assert en==old_names
            rows=(local[begin:end,None]*10+np.arange(10)).ravel()
            dest[rows,:344]=extended;dest[rows,344:]=memory[begin*10:end*10]
        dest.flush()
        checks.append({'period':period,'factor_sha256':digest(factor),'cut':cut,
                       'max_fit_timestamp':int(fit['timestamp'].max()),'query_target_invariance':True,
                       'pair_seen_fraction':float(np.mean(memory[:,0]>0)),
                       'events':len(ix),'seconds':time.time()-start})
        print(checks[-1],flush=True);del raw,memory,fit,query,ck;gc.collect()
    np.savez_compressed(a.out/'alignment.npz',**alignment)
    (a.out/'names.json').write_text(json.dumps(all_names)+'\n')
    # Check restored inputs by reproducing the released V20 neural OOF outputs.
    expected=np.load(a.v20_neural/'predictions.npz')['prediction']; restored=np.empty_like(expected)
    for fold in range(3):
        mask=np.flatnonzero(alignment['fold'][position]==fold)
        rows=(position[mask,None]*10+np.arange(10)).ravel();rr=(mask[:,None]*10+np.arange(10)).ravel()
        model=torch.load(a.v20_neural/f'fold_{fold}.pt',map_location='cpu',weights_only=False)
        restored[rr]=predict(model,dest[rows,:344])
    error=float(np.max(np.abs(restored-expected)))
    if error>1e-7:raise ValueError(f'Restored V20 neural predictions differ: {error}')
    report={'plan':plan,'checks':checks,'columns':len(all_names),'max_v20_neural_prediction_error':error,
            'feature_sha256':digest(a.out/'raw.npy'),'seconds':time.time()-start,
            'limits':'Reuses original Saturday/Sunday calibration cohorts; memory has no query-day outcomes.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
