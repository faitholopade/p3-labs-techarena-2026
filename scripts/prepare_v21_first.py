"""Build first-post-cutoff queries, including the second-day gap cohort."""
import argparse, gc, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np, torch, lightgbm as lgb
from common import load_data, subset, binary_labels, split_data
from daily_inference import build
from history_tree_features import extend
from flexible_tree import predict as tree_predict
from calibration_model import predict as neural_predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ['root', 'v13-extension', 'v17-extension', 'v20-extension', 'v20-research', 'out']:
        p.add_argument('--' + k, type=Path, required=True)
    p.add_argument('--threads', type=int, default=3)
    a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=True); torch.set_num_threads(a.threads)
    started=time.time(); data, evaluation, audit=load_data(a.root/'data')
    old=dict(np.load(a.v13_extension/'validation/v13/evidence.npz'))
    known=old['indices']; known_fold=old['fold'].reshape(-1,10)[:,0]
    user_fold=(np.arange(audit['n_users']+1,dtype=np.int64)*2654435761%3).astype(np.int8)
    for f in range(3):
        users=np.unique(data['user'][known[known_fold==f]])
        assert not np.intersect1d(users,data['user'][known[known_fold!=f]]).size
        user_fold[users]=f
    cuts=[1737763200,1737849600]
    selections=[split_data(data,cut)[1] for cut in cuts]
    indices=np.concatenate(selections); cutoffs=np.concatenate([np.full(len(ix),cut) for ix,cut in zip(selections,cuts)])
    lookup={(int(ix),int(cut)):pos for pos,(ix,cut) in enumerate(zip(indices,cutoffs))}
    known_cut=np.where(data['timestamp'][known]<cuts[1],cuts[0],cuts[1])
    matched=np.array([lookup[(int(ix),int(cut))] for ix,cut in zip(known,known_cut)])
    assert np.array_equal(user_fold[data['user'][indices[matched]]],known_fold)
    long_gap=data['timestamp'][indices]-cutoffs>=86400
    alignment=dict(indices=indices,cutoffs=cutoffs,user=data['user'][indices],fold=user_fold[data['user'][indices]],
                   labels=binary_labels(subset(data,indices)).ravel(),matched_position=matched,matched_indices=known,
                   period=(cutoffs==cuts[1]).astype(np.int8),long_gap=long_gap)
    np.savez_compressed(a.out/'alignment.npz',**alignment)
    parent=json.loads((a.v20_extension/'models/v20_spec.json').read_text())
    tree_spec=next(h['training_spec'] for h in parent['heads'] if h['name']=='expanded700')
    names=tree_spec['names']; (a.out/'names.json').write_text(json.dumps(names))
    dest=np.lib.format.open_memmap(a.out/'raw.npy',mode='w+',dtype=np.float32,shape=(len(indices)*10,len(names)))
    pos=0
    for period,cut,ix in zip(['earlier','later'],cuts,selections):
        query=subset(data,ix); query['target']=np.full(len(ix),-1,dtype=np.int32)
        assert np.all((query['hist']==0)|(query['hist_ts']<cut))
        np.savez_compressed(a.out/(period+'_query.npz'),**query)
        prior=subset(data,np.flatnonzero(data['timestamp']<cut))
        factor=a.v17_extension/f'validation/factors/{period}_factor_users_only.pt'
        print({'building':period,'events':len(ix),'long_gap_events':int((query['timestamp']-cut>=86400).sum())},flush=True)
        raw,raw_names=build(prior,query,cut,audit['n_items'],audit['n_users'],factor,state_root=a.v13_extension/'validation/source_states',posterior_samples=256)
        del prior; gc.collect(); state=torch.load(factor,map_location='cpu',weights_only=False)['state']
        for start in range(0,len(ix),4000):
            end=min(start+4000,len(ix)); x,ext_names=extend(raw[start*10:end*10],subset(query,np.arange(start,end)),state,raw_names)
            assert ext_names==names; dest[(pos+start)*10:(pos+end)*10]=x
        pos+=len(ix); dest.flush(); del raw,query,state,x; gc.collect()
    baseline={key:np.empty(len(indices)*10) for key in ['tree','neural']}
    for f in range(3):
        rows=(np.flatnonzero(alignment['fold']==f)[:,None]*10+np.arange(10)).ravel()
        model=lgb.Booster(model_file=str(a.v20_extension/f'validation/fold_models/expanded700_{f}.txt'))
        baseline['tree'][rows]=tree_predict(model,dest[rows],tree_spec)
        model=torch.load(a.v20_extension/f'validation/fold_models/neural32_{f}.pt',map_location='cpu',weights_only=False)
        baseline['neural'][rows]=neural_predict(model,dest[rows]); del model;gc.collect()
    r=(matched[:,None]*10+np.arange(10)).ravel(); parity={}
    for key,head in [('tree','expanded700'),('neural','neural32')]:
        expected=np.load(a.v20_research/head/'predictions.npz')['prediction']
        parity[key]=float(np.max(np.abs(baseline[key][r]-expected)))
        assert parity[key]<1e-6,parity
    np.savez_compressed(a.out/'baseline_heads.npz',**baseline)
    eval_gaps=(evaluation['timestamp']-1737936000)/86400
    report={'events':len(indices),'unique_outcome_events':int(len(np.unique(indices))),'matched_events':len(known),
            'long_gap_events':int(long_gap.sum()),'columns':len(names),'cutoffs':cuts,'posterior_samples':256,
            'query_labels_masked':True,'history_before_cutoff':True,'folds_exclude_users_across_cutoffs':True,
            'v20_head_parity':parity,'evaluation_events':len(eval_gaps),
            'evaluation_gap_ge_1day':int((eval_gaps>=1).sum()),'evaluation_gap_ge_2days':int((eval_gaps>=2).sum()),
            'evaluation_gap_ge_1day_fraction':float((eval_gaps>=1).mean()),
            'limits':'Long-gap cohort reuses later-day outcomes with an earlier cutoff; it is not an independent holdout. Gaps beyond two days remain unsupported. Only the two calibration heads are available for long-gap comparison.',
            'seconds':time.time()-started}
    (a.out/'spec.json').write_text(json.dumps(report,indent=2)); print(report,flush=True)


if __name__=='__main__': main()
