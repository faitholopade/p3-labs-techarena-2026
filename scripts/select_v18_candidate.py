"""Select a disclosed small tree-blend grid against the scored V17 incumbent."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ev=dict(np.load(a.incumbent));base=ev['candidate'];y=ev['labels'];preds={}
    for name in ['tree500','neural_tree200']:
        d=np.load(a.experiments/name/'predictions.npz')
        assert np.array_equal(d['baseline'],base) and np.array_equal(d['labels'],y) and np.array_equal(d['indices'],ev['indices'])
        preds[name]=d['prediction']
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    baseline={k:float(roc_auc_score(y[m],base[m])) for k,m in masks.items()};grid=[];values={}
    for members,shares in [(['tree500'],[1.]),(['neural_tree200'],[1.]),(['tree500','neural_tree200'],[.5,.5])]:
        branch=np.average([preds[m] for m in members],axis=0,weights=shares)
        for w in [.0625,.125,.25]:
            pred=(1-w)*base+w*branch;auc={k:float(roc_auc_score(y[m],pred[m])) for k,m in masks.items()};gain={k:auc[k]-baseline[k] for k in auc}
            name='+'.join(members)+f'@{w}';entry={'name':name,'members':members,'weights':[1-w]+[w*s for s in shares],'auc':auc,'gain':gain,'both_dates_improve':gain['earlier']>0 and gain['later']>0}
            grid.append(entry);values[name]=pred
    eligible=[g for g in grid if g['both_dates_improve'] and g['gain']['combined']>0]
    if not eligible:raise ValueError('No positive tree candidate on both dates.')
    best=max(g['gain']['combined'] for g in eligible);near=[g for g in eligible if g['gain']['combined']>=best-.000005]
    chosen=min(near,key=lambda g:(len(g['members']),1-g['weights'][0],-g['gain']['combined']))
    selection={**chosen,'baseline':'v17','official_auc':None,'selection_tolerance':.000005,
               'selection_rule':'Within 0.000005 of the best eligible combined gain, prefer fewer new heads, then the smaller new-head weight.',
               'release_status':'pending_assessment','authorization':'User explicitly authorised trying the strongest exploratory candidate below the normal threshold.',
               'limitation':'Nine blend comparisons on reused development cohorts; no public-score forecast.'}
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'evidence.npz',baseline=base,candidate=values[chosen['name']],labels=y,users=ev['users'],period=ev['period'],indices=ev['indices'])
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2));(a.out/'grid.json').write_text(json.dumps({'baseline':baseline,'candidates':grid,'selection':chosen['name']},indent=2))
    print(json.dumps({'grid':[{k:g[k] for k in ['name','gain','both_dates_improve']} for g in grid],'chosen':chosen['name']},indent=2))


if __name__=='__main__':main()
