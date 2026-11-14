"""Reconstruct one explicitly chosen V16 mixture and its paired validation evidence."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['experiments','incumbent','v13-evidence','legacy','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--members',nargs='+',required=True);p.add_argument('--screen',default='screen256')
    p.add_argument('--posterior-samples',type=int,default=256)
    p.add_argument('--anchor-alt',type=Path)
    p.add_argument('--branch',choices=['replacement','sequence'],required=True)
    p.add_argument('--weight',type=float,required=True)
    a=p.parse_args()
    if not 0<a.weight<=1:raise ValueError('Mixture weight must be in (0,1].')
    e=dict(np.load(a.incumbent));old=np.load(a.v13_evidence);legacy=np.load(a.legacy)
    assert np.array_equal(e['indices'],old['indices']) and np.array_equal(e['labels'],old['labels'])
    b=e['candidate'];parts=[]
    for member in a.members:
        s=np.full(len(b),np.nan)
        for period in ['earlier','later']:
            d=np.load(a.experiments/f'{period}_{member}'/a.screen/'predictions.npz')
            rr=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],e['labels'][rr]) and np.array_equal(d['baseline'],b[rr])
            s[rr]=d['sequence']
        assert np.isfinite(s).all();parts.append(s)
    factor_weights=[1/len(parts)]*len(parts);samples=[a.posterior_samples]*len(parts)
    members=list(a.members)
    if a.anchor_alt:
        alt=np.full(len(b),np.nan)
        for period in ['earlier','later']:
            d=np.load(a.anchor_alt/f'{period}_altinit/screen/predictions.npz')
            rr=(d['selection'][:,None]*10+np.arange(10)).ravel()
            assert np.array_equal(d['labels'],e['labels'][rr]);alt[rr]=d['sequence']
        assert np.isfinite(alt).all();parts.append(alt);factor_weights=[.75*x for x in factor_weights]+[.25]
        samples.append(64);members.append('incumbent_alt')
    seq=np.average(parts,axis=0,weights=factor_weights);w=a.weight
    weights=[1-w,.5*w,.1875*w,.3125*w] if a.branch=='replacement' else [1-w,0,.375*w,.625*w]
    candidate=weights[0]*b+weights[1]*old['v12']+weights[2]*legacy+weights[3]*seq
    e['baseline']=b;e['candidate']=candidate
    assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1
    assert candidate.reshape(-1,10).sum(1).max()<=1+1e-6
    a.out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out/'evidence.npz',**e)
    selection={'members':members,'factor_weights':factor_weights,'factor_posterior_samples':samples,
               'weights':weights,'branch':a.branch,'branch_weight':w,'screen':a.screen,
               'incumbent':'v15','incumbent_official_auc':.832366,'official_auc':None,
               'authorization':'The user asked to continue until a candidate and explicitly authorised an exploratory attempt if it remained below the normal threshold.',
               'limitation':'Selected using reused development cohorts. Public improvement is unverified.'}
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2));print(json.dumps(selection,indent=2))


if __name__=='__main__':main()
