"""Combine completed factor/head seed checks on the fixed V13 development rows."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--experiments',required=True);a=p.parse_args();exp=Path(a.experiments)
    e=dict(np.load(Path(a.extension)/'validation/v13/evidence.npz'));y=e['labels'];base=e['v13']
    legacy=np.load(exp/'cache/legacy.npy');period=np.empty(len(y),np.int8);sequences=[]
    for suffix in ['', '_seed2']:
        pred=np.full(len(y),np.nan)
        for i,name in enumerate(['earlier','later']):
            d=np.load(exp/f'{name}_prior05_warm2_rank05{suffix}/screen/predictions.npz')
            rows=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],y[rows]) and np.array_equal(d['baseline'],base[rows])
            pred[rows]=d['sequence'];period[rows]=i
        assert np.isfinite(pred).all();sequences.append(pred)
    frozen=np.mean(sequences,axis=0);adapted=[]
    for seed in [731,932,1417]:
        d=np.load(exp/f'adaptation{seed}/predictions.npz')
        assert np.array_equal(d['labels'],y) and np.array_equal(d['indices'],e['indices'])
        adapted.append(d['sequence'])
    adapted=np.mean(adapted,axis=0);results=[];candidates={}
    for weight in [0.,.25,.5,.75,1.]:
        seq=(1-weight)*frozen+weight*adapted
        prediction=.5*e['v12']+.1875*legacy+.3125*seq;candidates[weight]=prediction
        row={'adaptation_weight':weight,'cohorts':{}}
        for name,mask in [('combined',np.ones(len(y),bool)),('earlier',period==0),('later',period==1)]:
            auc=float(roc_auc_score(y[mask],prediction[mask]));baseline_auc=float(roc_auc_score(y[mask],base[mask]))
            row['cohorts'][name]={'auc':auc,'baseline_auc':baseline_auc,'gain':auc-baseline_auc,'rows':int(mask.sum())}
        results.append(row)
    selected=max(results,key=lambda x:x['cohorts']['combined']['auc']);weight=selected['adaptation_weight']
    d=np.load(Path(a.root)/'data/train.npz');users=np.repeat(d['user'][e['indices']],10)
    out=exp/'assessment';out.mkdir(exist_ok=True)
    np.savez_compressed(out/'evidence.npz',candidate=candidates[weight],baseline=base,labels=y,users=users,period=period,indices=e['indices'])
    report={'results':results,'selected':selected,'selection_limitation':'Blend and model selection reuse the development cohorts; this is not an independent test.',
            'formula':'0.5*V12 + 0.1875*V8 + 0.3125*((1-weight)*mean_two_factor_seeds_frozen_heads + weight*mean_three_adapted_heads)',
            'factor_seeds':[20261027,20261028],'head_seeds':[731,932,1417]}
    (out/'selection.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
