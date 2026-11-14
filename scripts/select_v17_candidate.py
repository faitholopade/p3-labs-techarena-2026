"""Compare a small, disclosed V17 blend grid on matched development rows."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ev=dict(np.load(a.incumbent));base=ev['candidate'];y=ev['labels'];period=ev['period']
    sequences={};legacy=np.load(a.experiments/'cache/legacy.npy')
    for member,screen in [('cold_kl1','screen'),('users_only','screen256')]:
        sequence=np.empty(len(base));seen=np.zeros(len(base),bool)
        for name in ['earlier','later']:
            d=np.load(a.experiments/f'{name}_{member}/{screen}/predictions.npz')
            rows=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['baseline'],base[rows]) and np.array_equal(d['labels'],y[rows])
            assert not seen[rows].any();sequence[rows]=d['sequence'];seen[rows]=True
        assert seen.all();sequences[member]=sequence
    baseline={name:float(roc_auc_score(y[mask],base[mask])) for name,mask in [('combined',np.ones(len(y),bool)),('earlier',period==0),('later',period==1)]}
    reports=[];predictions={}
    for members,fw in [(['cold_kl1'],[1.]),(['users_only'],[1.]),(['cold_kl1','users_only'],[.5,.5])]:
        sequence=np.average([sequences[x] for x in members],axis=0,weights=fw)
        branch=.375*legacy+.625*sequence
        for blend in [.125,.25]:
            name='+'.join(members)+f'@{blend}';pred=(1-blend)*base+blend*branch
            auc={name:float(roc_auc_score(y[mask],pred[mask])) for name,mask in [('combined',np.ones(len(y),bool)),('earlier',period==0),('later',period==1)]}
            gain={name:auc[name]-baseline[name] for name in auc}
            reports.append({'name':name,'members':members,'factor_weights':fw,'weights':[1-blend,0,.375*blend,.625*blend],
                            'factor_posterior_samples':[256]*len(members),'auc':auc,'gain':gain,
                            'both_dates_improve':gain['earlier']>0 and gain['later']>0})
            predictions[name]=pred
    eligible=[x for x in reports if x['both_dates_improve'] and x['gain']['combined']>0]
    if not eligible:raise ValueError('No positive candidate on both dates.')
    chosen=max(eligible,key=lambda x:x['gain']['combined'])
    chosen={**chosen,'baseline':'v16','official_auc':None,'release_status':'pending_paired_assessment',
            'authorization':'User explicitly authorised continuing until a candidate, and trying the strongest exploratory candidate below the usual threshold.',
            'limitations':'Six additional blend comparisons on reused development cohorts. Bootstrap is conditional on selection; no estimate of public rank.'}
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'evidence.npz',baseline=base,candidate=predictions[chosen['name']],labels=y,users=ev['users'],period=period,indices=ev['indices'])
    (a.out/'selection.json').write_text(json.dumps(chosen,indent=2))
    (a.out/'grid.json').write_text(json.dumps({'baseline':baseline,'candidates':reports,'selection':chosen['name']},indent=2))
    print(json.dumps({'candidates':[{k:x[k] for k in ['name','gain','both_dates_improve']} for x in reports],'selected':chosen['name']},indent=2))


if __name__=='__main__':main()
