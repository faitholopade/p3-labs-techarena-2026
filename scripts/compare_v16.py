"""Compare fixed V16 factor branches against the officially scored V15 model.

Each requested member must have earlier/later frozen user-excluded head screens.
Preserve every comparison, including the unchanged V15 branch-weight control.
"""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiments',type=Path,required=True)
    p.add_argument('--incumbent',type=Path,required=True)
    p.add_argument('--v13-evidence',type=Path,required=True)
    p.add_argument('--legacy',type=Path,required=True)
    p.add_argument('--members',nargs='+',required=True)
    p.add_argument('--screen',default='screen')
    p.add_argument('--anchor-alt',type=Path,help='V15 follow-up experiments containing earlier/later_altinit/screen predictions.')
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    ev=dict(np.load(a.incumbent));old=np.load(a.v13_evidence)
    assert np.array_equal(ev['indices'],old['indices'])
    assert np.array_equal(ev['labels'],old['labels'])
    b=ev['candidate'];v12=old['v12'];legacy=np.load(a.legacy)
    incumbent_sequence=(b-.5*v12-.1875*legacy)/.3125
    sequences={'unchanged_v15':incumbent_sequence}
    for name in a.members:
        pred=np.full(len(b),np.nan)
        for period in ['earlier','later']:
            d=np.load(a.experiments/f'{period}_{name}'/a.screen/'predictions.npz')
            rr=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],ev['labels'][rr])
            assert np.array_equal(d['baseline'],b[rr])
            pred[rr]=d['sequence']
        assert np.isfinite(pred).all();sequences[name]=pred
    if len(a.members)>1:sequences['equal_mean']=np.mean([sequences[x] for x in a.members],axis=0)
    if a.anchor_alt:
        alt=np.full(len(b),np.nan)
        for period in ['earlier','later']:
            d=np.load(a.anchor_alt/f'{period}_altinit/screen/predictions.npz')
            rr=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],ev['labels'][rr]);alt[rr]=d['sequence']
        assert np.isfinite(alt).all()
        sequences['retain_v15_alt']=.75*np.mean([sequences[x] for x in a.members],axis=0)+.25*alt
    masks={'combined':np.ones(len(b),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    baseline={k:float(roc_auc_score(ev['labels'][m],b[m])) for k,m in masks.items()}
    results=[]
    for name,seq in sequences.items():
        for branch,weight_list in [('replacement',[.25,.5,.75,1.]),('sequence',[.25,.5])]:
            target=.5*v12+.1875*legacy+.3125*seq if branch=='replacement' else .375*legacy+.625*seq
            for weight in weight_list:
                key=f'{name}_{branch}_{weight:g}';candidate=(1-weight)*b+weight*target
                auc={k:float(roc_auc_score(ev['labels'][m],candidate[m])) for k,m in masks.items()}
                gains={k:auc[k]-baseline[k] for k in auc}
                results.append({'key':key,'member':name,'branch':branch,'weight':weight,'auc':auc,'gain':gains})
    results.sort(key=lambda r:r['gain']['combined'],reverse=True)
    report={'baseline_version':'v15','baseline':baseline,'members':a.members,'screen':a.screen,
            'limitation':'Reused development cohorts; comparison is conditional on model and blend selection.','results':results}
    (a.out/'comparison.json').write_text(json.dumps(report,indent=2))
    # Reconstruct and save only the explicitly selected candidate after review.
    print(json.dumps(results[:12],indent=2))


if __name__=='__main__':main()
