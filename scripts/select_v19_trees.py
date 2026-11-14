"""Compare a disclosed final grid of richer tree mixtures against V18."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','old-tree','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ev=dict(np.load(a.incumbent));base=ev['candidate'];v17=ev['baseline'];y=ev['labels'];old=np.load(a.old_tree)['prediction'];assert np.max(abs(.875*v17+.125*old-base))<1e-12
    preds={}
    for name in ['event15','binary15','event7','rank15']:
        d=np.load(a.experiments/name/'predictions.npz');assert np.array_equal(d['baseline'],base) and np.array_equal(d['labels'],y) and np.array_equal(d['indices'],ev['indices']);preds[name]=d['prediction']
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1};auc0={k:float(roc_auc_score(y[m],base[m])) for k,m in masks.items()}
    branches=[[x] for x in preds]+[['event15','event7'],['event15','rank15'],['event7','rank15'],list(preds)]
    grid=[];values={}
    for members in branches:
        branch=np.mean([preds[x] for x in members],axis=0)
        for mode,amount in [('replace',.0625),('replace',.125),('add',.0625),('add',.125),('add',.25)]:
            if mode=='replace':weights=[.875,.125-amount]+[amount/len(members)]*len(members)
            else:weights=[.875*(1-amount),.125*(1-amount)]+[amount/len(members)]*len(members)
            pred=weights[0]*v17+weights[1]*old+sum(w*preds[x] for w,x in zip(weights[2:],members))
            auc={k:float(roc_auc_score(y[m],pred[m])) for k,m in masks.items()};gain={k:auc[k]-auc0[k] for k in auc};name='+'.join(members)+f'/{mode}@{amount}'
            entry={'name':name,'members':members,'weights':weights,'weight_order':['v17','v18_tree']+members,'mode':mode,'new_weight':amount,'auc':auc,'gain':gain,'both_dates_improve':gain['earlier']>0 and gain['later']>0};grid.append(entry);values[name]=pred
    eligible=[x for x in grid if x['both_dates_improve'] and x['gain']['combined']>0]
    if not eligible:raise ValueError('No positive candidate on both dates.')
    best=max(x['gain']['combined'] for x in eligible);near=[x for x in eligible if x['gain']['combined']>=best-.000005]
    chosen=min(near,key=lambda x:(len(x['members']),x['new_weight'],-x['gain']['combined']))
    selection={**chosen,'baseline':'v18','official_auc':None,'status':'pending_final_factor_screen_and_assessment','selection_tolerance':.000005,
               'selection_rule':'Among candidates with both dates positive, within 0.000005 of best combined gain, prefer fewer new heads, then smaller new-head weight.',
               'authorization':'User explicitly authorised trying the strongest exploratory candidate below the normal threshold.',
               'limitations':'40 final tree comparisons, plus initial screens and earlier research, on reused development cohorts; no public-score forecast.'}
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'evidence.npz',baseline=base,candidate=values[chosen['name']],labels=y,users=ev['users'],period=ev['period'],indices=ev['indices'])
    (a.out/'selection.json').write_text(json.dumps(selection,indent=2));(a.out/'grid.json').write_text(json.dumps({'baseline':auc0,'candidates':grid,'selection':chosen['name']},indent=2))
    print(json.dumps({'selected':selection,'best_eligible_gain':best,'grid_count':len(grid)},indent=2),flush=True)


if __name__=='__main__':main()
