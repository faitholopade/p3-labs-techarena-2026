"""Compare independent-start diversity and higher-precision posterior integration."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--extension',required=True)
    p.add_argument('--previous',required=True);p.add_argument('--experiments',required=True);a=p.parse_args()
    old=Path(a.previous);new=Path(a.experiments);e=dict(np.load(Path(a.extension)/'validation/v13/evidence.npz'))
    y=e['labels'];base=e['v13'];legacy=np.load(old/'cache/legacy.npy');period=np.empty(len(y),np.int8)
    families={'original1':{},'original2':{},'qmc1':{},'qmc2':{},'alt':{}}
    for label in ['earlier','later']:
        families['original1'][label]=old/f'{label}_prior05_warm2_rank05/screen/predictions.npz'
        families['original2'][label]=old/f'{label}_prior05_warm2_rank05_seed2/screen/predictions.npz'
        families['qmc1'][label]=new/f'{label}_qmc256_seed1/predictions.npz'
        families['qmc2'][label]=new/f'{label}_qmc256_seed2/predictions.npz'
        families['alt'][label]=new/f'{label}_altinit/screen/predictions.npz'
    seq={}
    for name,paths in families.items():
        s=np.full(len(y),np.nan)
        for i,label in enumerate(['earlier','later']):
            d=np.load(paths[label]);rows=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],y[rows]) and np.array_equal(d['baseline'],base[rows])
            s[rows]=d['sequence'];period[rows]=i
        assert np.isfinite(s).all();seq[name]=s
    definitions={
        'two_seed_64':(['original1','original2'],[.5,.5],[64,64]),
        'two_seed_256':(['qmc1','qmc2'],[.5,.5],[256,256]),
        'alt_only':(['alt'],[1.],[64]),
        'alt25_64':(['original1','original2','alt'],[.375,.375,.25],[64,64,64]),
        'alt50_64':(['original1','original2','alt'],[.25,.25,.5],[64,64,64]),
        'alt25_256':(['qmc1','qmc2','alt'],[.375,.375,.25],[256,256,64]),
        'alt50_256':(['qmc1','qmc2','alt'],[.25,.25,.5],[256,256,64]),
    }
    reports=[];predictions={}
    for name,(members,weights,samples) in definitions.items():
        s=np.average([seq[k] for k in members],weights=weights,axis=0)
        pred=.5*e['v12']+.1875*legacy+.3125*s;predictions[name]=pred
        row={'name':name,'members':members,'factor_weights':weights,'factor_posterior_samples':samples,'cohorts':{}}
        for date,mask in [('combined',np.ones(len(y),bool)),('earlier',period==0),('later',period==1)]:
            auc=float(roc_auc_score(y[mask],pred[mask]));old_auc=float(roc_auc_score(y[mask],base[mask]))
            row['cohorts'][date]={'auc':auc,'baseline_auc':old_auc,'gain':auc-old_auc,'rows':int(mask.sum())}
        reports.append(row)
    eligible=[row for row in reports if all(row['cohorts'][k]['gain']>0 for k in ['earlier','later'])]
    selected=max(eligible,key=lambda row:row['cohorts']['combined']['auc'])
    data=np.load(Path(a.root)/'data/train.npz');users=np.repeat(data['user'][e['indices']],10)
    dest=new/'assessment';dest.mkdir(exist_ok=True)
    np.savez_compressed(dest/'evidence.npz',candidate=predictions[selected['name']],baseline=base,labels=y,
                        users=users,period=period,indices=e['indices'])
    result={'selection':selected,'candidates':reports,'user_authorized_exploratory_release':True,
            'authorization':'Keep going until a candidate; if none clears the threshold, try the strongest validated candidate anyway.',
            'formula':'0.5*V12 + 0.1875*V8 + 0.3125*weighted_new_factor_sequence_probability',
            'limitations':'Development rows reused for model and blend selection. No hidden evaluation outcomes used. AUC here is local, not official.'}
    (dest/'selection.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)


if __name__=='__main__':main()
