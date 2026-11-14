"""Build two complete weekday calibration cohorts with fresh matching factors."""
import argparse, gc, json, shutil, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from common import load_data, subset, binary_labels
from daily_inference import build
from history_tree_features import extend
from v25_exposure_memory import features as memory_features
from backtest_v24_factor import sha256, write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','factors','fresh-bases','v20-features','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(3)
    data,_,audit=load_data(a.root/'data');old=dict(np.load(a.v20_features/'alignment.npz'))
    names=json.loads((a.v20_features/'names.json').read_text())[:344]
    assignment=(np.arange(audit['n_users']+1,dtype=np.int64)*2654435761%3).astype(np.int8)
    for fold in range(3):assignment[np.unique(old['user'][old['fold']==fold])]=fold
    for day,cut in [('thursday',1737590400),('friday',1737676800)]:
        folder=a.out/day;folder.mkdir(exist_ok=True)
        if (folder/'report.json').exists():
            report=json.loads((folder/'report.json').read_text());assert sha256(folder/'raw.npy')==report['feature_sha256'];continue
        if (folder/'raw.npy').exists():raise ValueError('Unverified partial feature file.')
        factor=a.factors/day/'users/residual.pt';ck=torch.load(factor,map_location='cpu',weights_only=False)
        assert ck['cut']==cut
        completed=json.loads((a.factors/day/'users/completed.json').read_text());assert sha256(factor)==completed['checkpoint_sha256']
        ix=np.flatnonzero((data['timestamp']>=cut)&(data['timestamp']<cut+86400))
        query=subset(data,ix);labels=binary_labels(query).ravel();query['target']=np.full(len(ix),-1)
        fit=subset(data,np.flatnonzero(data['timestamp']<cut));assert fit['timestamp'].max()<cut
        bases=a.out/'bases'/str(cut);bases.mkdir(parents=True,exist_ok=True)
        for file in ['exposure.npz','p0_u0.5_i0.5_r5.npy']:
            shutil.copyfile(a.fresh_bases/(day+'_factor')/'bases'/file,bases/file)
        started=time.time();print({'stage':'prepare_weekday','day':day,'events':len(ix)},flush=True)
        raw,base_names=build(fit,query,cut,audit['n_items'],audit['n_users'],factor,state_root=a.out/'bases',posterior_samples=256)
        memory,memory_names=memory_features(fit,query,ck['state'],cut)
        dest=np.lib.format.open_memmap(folder/'raw.npy',mode='w+',dtype=np.float32,shape=(len(ix)*10,344+len(memory_names)))
        for start in range(0,len(ix),4000):
            stop=min(start+4000,len(ix));value,new_names=extend(raw[start*10:stop*10],subset(query,np.arange(start,stop)),ck['state'],base_names)
            assert new_names==names;dest[start*10:stop*10,:344]=value
            dest[start*10:stop*10,344:]=memory[start*10:stop*10]
        dest.flush();del raw,dest,memory;gc.collect()
        _,first=np.unique(query['user'],return_index=True);first=np.sort(first)
        # Original arrays are chronological; explicitly verify first-event cohorts.
        for field in ['timestamp']:
            assert np.all(np.diff(query[field])>=0)
        np.savez_compressed(folder/'alignment.npz',indices=ix,user=query['user'],labels=labels,
                            fold=assignment[query['user']],first_position=first,timestamp=query['timestamp'])
        write_json(folder/'names.json',names+memory_names)
        report={'cut':cut,'events':len(ix),'first_events':len(first),'factor_sha256':sha256(factor),
                'feature_sha256':sha256(folder/'raw.npy'),'max_fit_timestamp':int(fit['timestamp'].max()),
                'query_labels_masked':True,'columns':344+len(memory_names),'seconds':time.time()-started,
                'source_sha256':sha256(Path(__file__)),'scope':'V17 users-only family; no complete V20 weekday ensemble.'}
        write_json(folder/'report.json',report);print(report,flush=True)
        del query,fit,ck;gc.collect()

if __name__=='__main__':main()
