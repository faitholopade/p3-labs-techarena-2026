"""Prepare full daily, chronological features for factor-head adaptation."""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from common import load_data,subset,binary_labels
from daily_inference import build
from sequence_calibration import history_pairs


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--experiments',required=True);p.add_argument('--family',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();root=Path(a.root);extension=Path(a.extension);exp=Path(a.experiments);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data,_,audit=load_data(root/'data');e=np.load(extension/'validation/v13/evidence.npz')
    folds=e['fold'].reshape(-1,10)[:,0];users=data['user'][e['indices']]
    user_fold=(np.arange(audit['n_users']+1,dtype=np.int64)*2654435761%3).astype(np.int8)
    known={}
    for user,fold in zip(users,folds):
        if int(user) in known:assert known[int(user)]==int(fold)
        known[int(user)]=int(fold);user_fold[user]=fold
    np.save(out/'user_fold.npy',user_fold)
    for period,cut in [('earlier',1737763200),('later',1737849600)]:
        dest=out/period;dest.mkdir(exist_ok=True)
        indices=np.flatnonzero((data['timestamp']>=cut)&(data['timestamp']<cut+86400))
        query=subset(data,indices);prior=subset(data,np.flatnonzero(data['timestamp']<cut))
        factor=exp/f'{period}_{a.family}'/'residual.pt'
        if not (dest/'raw.npy').exists():
            raw,names=build(prior,query,cut,audit['n_items'],audit['n_users'],factor,state_root=extension/'validation/source_states')
            np.save(dest/'raw.npy',raw);(dest/'names.json').write_text(json.dumps(names));del raw
        del prior;gc.collect()
        state=torch.load(factor,map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,query)
        np.save(dest/'pairs.npy',pairs);np.save(dest/'mask.npy',mask)
        np.savez_compressed(dest/'alignment.npz',indices=indices,user=query['user'],fold=user_fold[query['user']],labels=binary_labels(query).ravel())
        print({'prepared':period,'events':len(indices)},flush=True)
        del pairs,query,state;gc.collect()
    (out/'spec.json').write_text(json.dumps({'family':a.family,'known_fold_users':len(known),'args':vars(a)},indent=2))


if __name__=='__main__':main()
