"""Adapt a history head to changed factors without crossing user folds."""
import argparse,json,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from sequence_training import fit
from sequence_calibration import predict


def main():
    p=argparse.ArgumentParser();p.add_argument('--extension',required=True);p.add_argument('--cache',required=True)
    p.add_argument('--out',required=True);p.add_argument('--seed',type=int,default=731);p.add_argument('--epochs',type=int,default=6)
    a=p.parse_args();extension=Path(a.extension);cache=Path(a.cache);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    e=np.load(extension/'validation/v13/evidence.npz');ds=[];align=[]
    for period in ['earlier','later']:
        d=cache/period;info=dict(np.load(d/'alignment.npz'));align.append(info)
        ds.append({'x':np.load(d/'raw.npy',mmap_mode='r'),'names':json.loads((d/'names.json').read_text()),
                   'labels':info['labels'],'pairs':np.load(d/'pairs.npy',mmap_mode='r'),'mask':np.load(d/'mask.npy'),
                   'user':info['user']})
    prediction=np.full(len(e['labels']),np.nan);folds=e['fold'].reshape(-1,10)[:,0]
    report={'args':vars(a),'folds':[]}
    for fold in range(3):
        old=torch.load(extension/'validation/v13/fold_models'/f'head_{fold}_{a.seed}.pt',map_location='cpu',weights_only=False)
        masks=[info['fold']!=fold for info in align]
        heldout=np.concatenate([info['user'][info['fold']==fold] for info in align])
        trained=np.concatenate([d['user'][mask] for d,mask in zip(ds,masks)])
        assert len(np.intersect1d(trained,heldout))==0
        ck=fit(old['base_spec'],ds,masks,a.seed+201,epochs=a.epochs,initial_sequence=old,base_lr=.0001,sequence_lr=.0002)
        torch.save(ck,out/f'head_{fold}_{a.seed}.pt')
        for d,info in zip(ds,align):
            selected=np.flatnonzero((folds==fold)&np.isin(e['indices'],info['indices']))
            local=np.searchsorted(info['indices'],e['indices'][selected]);assert np.array_equal(info['indices'][local],e['indices'][selected])
            rows=(local[:,None]*10+np.arange(10)).ravel();dst=(selected[:,None]*10+np.arange(10)).ravel()
            prediction[dst]=predict(ck,d['x'][rows],d['pairs'][local],d['mask'][local])
        report['folds'].append({'fold':fold,'curve':ck['curve'],'training_events':ck['training_events']})
        del ck,old;gc.collect();print({'completed_fold':fold},flush=True)
    assert np.isfinite(prediction).all()
    np.savez_compressed(out/'predictions.npz',sequence=prediction,indices=e['indices'],labels=e['labels'])
    report['raw_head_auc']=float(roc_auc_score(e['labels'],prediction))
    (out/'report.json').write_text(json.dumps(report,indent=2));print({'raw_head_auc':report['raw_head_auc']},flush=True)


if __name__=='__main__':main()
