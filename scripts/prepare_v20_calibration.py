"""Expand tree training days while preserving V19 features and user-held-out folds."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from common import load_data,subset,binary_labels
from daily_inference import build
from history_tree_features import extend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','v13-extension','v17-extension','matched-features','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();torch.set_num_threads(a.threads);a.out.mkdir(parents=True,exist_ok=True)
    data,_,audit=load_data(a.root/'data');e=np.load(a.v13_extension/'validation/v13/evidence.npz')
    known_indices=e['indices'];fold=e['fold'].reshape(-1,10)[:,0];known_users=data['user'][known_indices]
    user_fold=(np.arange(audit['n_users']+1,dtype=np.int64)*2654435761%3).astype(np.int8)
    for f in range(3):
        users=np.unique(known_users[fold==f]);assert not np.intersect1d(users,known_users[fold!=f]).size
        user_fold[users]=f
    indices=np.flatnonzero((data['timestamp']>=1737763200)&(data['timestamp']<1737936000))
    matched_position=np.searchsorted(indices,known_indices);assert np.array_equal(indices[matched_position],known_indices)
    target=binary_labels(subset(data,indices)).ravel();all_fold=user_fold[data['user'][indices]]
    np.savez_compressed(a.out/'alignment.npz',indices=indices,user=data['user'][indices],fold=all_fold,labels=target,
                        matched_position=matched_position,matched_indices=known_indices,period=(data['timestamp'][indices]>=1737849600).astype(np.int8))
    old=np.load(a.matched_features/'raw.npy',mmap_mode='r');old_names=json.loads((a.matched_features/'names.json').read_text())
    dest=np.lib.format.open_memmap(a.out/'raw.npy',mode='w+',dtype=np.float32,shape=(len(indices)*10,len(old_names)))
    checks=[];started=time.time()
    for period,cut in [('earlier',1737763200),('later',1737849600)]:
        local=np.flatnonzero((data['timestamp'][indices]>=cut)&(data['timestamp'][indices]<cut+86400))
        query=subset(data,indices[local]);query['target']=np.full(len(local),-1,dtype=np.int32)
        prior=subset(data,np.flatnonzero(data['timestamp']<cut));factor=a.v17_extension/f'validation/factors/{period}_factor_users_only.pt'
        print({'building':period,'events':len(local),'elapsed_seconds':time.time()-started},flush=True)
        raw,names=build(prior,query,cut,audit['n_items'],audit['n_users'],factor,
                        state_root=a.v13_extension/'validation/source_states',posterior_samples=256)
        del prior;gc.collect();state=torch.load(factor,map_location='cpu',weights_only=False)['state']
        for start in range(0,len(local),4000):
            end=min(start+4000,len(local));ii=np.arange(start,end)
            x,extended_names=extend(raw[start*10:end*10],subset(query,ii),state,names)
            assert extended_names==old_names
            dest[local[start]*10:(local[end-1]+1)*10]=x
        dest.flush();selection=np.flatnonzero((data['timestamp'][known_indices]>=cut)&(data['timestamp'][known_indices]<cut+86400))
        maxerr=0.;nan_equal=True
        for start in range(0,len(selection),1000):
            take=selection[start:start+1000];r=(take[:,None]*10+np.arange(10)).ravel()
            rr=(matched_position[take,None]*10+np.arange(10)).ravel();new=dest[rr];previous=old[r]
            nan_equal &= np.array_equal(np.isnan(new),np.isnan(previous));maxerr=max(maxerr,float(np.nanmax(np.abs(new-previous))))
        check={'period':period,'events':len(local),'matched_events':len(selection),'max_abs_feature_difference':maxerr,'nan_patterns_equal':bool(nan_equal)}
        checks.append(check);print(check,flush=True)
        if not nan_equal or maxerr>2e-5:raise ValueError('Expanded feature parity failed.')
        del raw,query,state,x;gc.collect()
    (a.out/'names.json').write_text(json.dumps(old_names))
    spec={'parent':'v19','events':len(indices),'rows':len(target),'columns':len(old_names),'posterior_samples':256,
          'feature_cutoffs':[1737763200,1737849600],'query_labels_masked':True,'folds_exclude_users_across_dates':True,
          'matched_events':len(known_indices),'fold_assignment':'Preserve V13 users; deterministic modulo hash for other users.',
          'parity':checks,'feature_function':'daily_inference.build + history_tree_features.extend','elapsed_seconds':time.time()-started}
    (a.out/'spec.json').write_text(json.dumps(spec,indent=2));print(spec,flush=True)


if __name__=='__main__':main()
