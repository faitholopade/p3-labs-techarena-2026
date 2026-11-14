"""Choose a disclosed, small mixture grid using original and long-gap checks."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--members',nargs='+',default=['first_neural','first_neural_set','first_neural_ce'])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);ev=dict(np.load(a.incumbent));al=dict(np.load(a.experiments/'features/alignment.npz'));old=dict(np.load(a.experiments/'features/baseline_heads.npz'))
    predictions={name:np.load(a.experiments/name/'predictions.npz')['prediction'] for name in a.members}
    r=(al['matched_position'][:,None]*10+np.arange(10)).ravel();long=np.repeat(al['long_gap'],10);base=ev['candidate']
    assert np.array_equal(al['labels'][r],ev['labels'])
    masks={'combined':np.ones(len(base),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    base_auc={k:float(roc_auc_score(ev['labels'][m],base[m])) for k,m in masks.items()};small=(7*old['tree']+8*old['neural'])/15
    small_auc=float(roc_auc_score(al['labels'][long],small[long]));grid=[]
    for n in range(1,len(a.members)+1):
        for members in itertools.combinations(a.members,n):
            avg=np.mean([predictions[k] for k in members],axis=0)
            for fraction in [.5,1.]:
                weight=.125*fraction;candidate=base+weight*(avg[r]-old['neural'][r])
                auc={k:float(roc_auc_score(ev['labels'][m],candidate[m])) for k,m in masks.items()};gain={k:auc[k]-base_auc[k] for k in masks}
                gap=float(roc_auc_score(al['labels'][long],(small+fraction*8/15*(avg-old['neural']))[long])-small_auc)
                grid.append({'members':list(members),'fraction_replaced':fraction,'effective_weight':weight,'auc':auc,'gain':gain,'long_gap_calibration_subensemble_gain':gap,'both_dates_improve':all(gain[k]>0 for k in ['earlier','later'])})
    eligible=[x for x in grid if x['both_dates_improve'] and x['long_gap_calibration_subensemble_gain']>0]
    (a.out/'grid.json').write_text(json.dumps(grid,indent=2))
    if not eligible:raise ValueError('No candidate improves both original dates and the long-gap calibration subensemble.')
    best=max(x['gain']['combined'] for x in eligible)
    selected=min([x for x in eligible if x['gain']['combined']>=best-.000002],key=lambda x:(len(x['members']),x['effective_weight'],-x['gain']['combined']))
    avg=np.mean([predictions[k] for k in selected['members']],axis=0);candidate=base+selected['effective_weight']*(avg[r]-old['neural'][r])
    np.savez_compressed(a.out/'evidence.npz',baseline=base,candidate=candidate,labels=ev['labels'],users=ev['users'],period=ev['period'],indices=ev['indices'])
    selected.update(version='v21',baseline='v20',official_auc=None,grid_count=len(grid),selection_tolerance=.000002,
        selection_rule='Both original dates positive and long-gap calibration subensemble positive; within 0.000002 of best combined gain prefer fewer heads, then lower replacement weight.',
        authorization='User explicitly authorised trying the strongest exploratory candidate below the normal threshold.',
        limitations='Repeated development reuse; the long-gap check reuses outcomes and only compares the calibration subensemble. No gain forecast for the public leaderboard.')
    (a.out/'selection.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2),flush=True)


if __name__=='__main__':main()
