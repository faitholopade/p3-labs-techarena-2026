"""Evaluate a disclosed small grid of expanded-data heads against scored V19."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','incumbent','v17-evidence','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();ev=dict(np.load(a.incumbent));v17=np.load(a.v17_evidence)['candidate'];y=ev['labels'];base=ev['candidate']
    components={'v19':base,'v17':v17}
    for member in ['expanded700','neural32']:
        d=np.load(a.experiments/member/'predictions.npz')
        assert np.array_equal(d['baseline'],base) and np.array_equal(d['labels'],y) and np.array_equal(d['indices'],ev['indices'])
        components[member]=d['prediction']
    order=list(components);recipes=[]
    for w in [.0625,.125,.25]:
        recipes.append((f'expanded_tree_{w}',np.array([0,1-w,w,0.])))
        recipes.append((f'half_replacement_{w}',np.array([.5,.5*(1-w),.5*w,0.])))
    for w in [.03125,.0625,.125,.25]:recipes.append((f'neural_{w}',np.array([1-w,0.,0.,w])))
    for label,recipe in list(recipes):
        if label not in ['expanded_tree_0.125','expanded_tree_0.25','half_replacement_0.125']:continue
        for w in [.03125,.0625,.125]:
            weights=recipe*(1-w);weights[3]=w;recipes.append((label+f'_neural_{w}',weights))
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    auc0={k:float(roc_auc_score(y[m],base[m])) for k,m in masks.items()};grid=[];predictions={}
    for name,weights in recipes:
        assert np.all(weights>=0) and np.isclose(weights.sum(),1)
        pred=sum(float(w)*components[k] for w,k in zip(weights,order));aucs={k:float(roc_auc_score(y[m],pred[m])) for k,m in masks.items()}
        gain={k:aucs[k]-auc0[k] for k in aucs};members=[k for k,w in zip(order[2:],weights[2:]) if w>0]
        entry={'name':name,'weights':weights.tolist(),'weight_order':order,'members':members,'auc':aucs,'gain':gain,
               'both_dates_improve':gain['earlier']>0 and gain['later']>0,'new_weight':float(weights[2:].sum())}
        grid.append(entry);predictions[name]=pred
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'grid.json').write_text(json.dumps({'baseline':auc0,'candidates':grid},indent=2))
    eligible=[x for x in grid if x['both_dates_improve'] and x['gain']['combined']>0]
    if not eligible:raise ValueError('No positive candidate on both dates.')
    best=max(x['gain']['combined'] for x in eligible);near=[x for x in eligible if x['gain']['combined']>=best-.000005]
    chosen=min(near,key=lambda x:(len(x['members']),x['new_weight'],-x['gain']['combined']))
    chosen.update({'baseline':'v19','official_auc':None,'status':'pending_paired_assessment','grid_count':len(grid),'selection_tolerance':.000005,
                   'selection_rule':'Both dates positive; within 0.000005 of best combined gain prefer fewer new heads, then smaller total new-head weight.',
                   'authorization':'User explicitly authorised trying the strongest exploratory candidate below the normal threshold.',
                   'limitations':'Two new head families and 19 final mixtures on repeatedly reused development cohorts; no public-score forecast.'})
    np.savez_compressed(a.out/'evidence.npz',baseline=base,candidate=predictions[chosen['name']],labels=y,users=ev['users'],period=ev['period'],indices=ev['indices'])
    (a.out/'selection.json').write_text(json.dumps(chosen,indent=2));print(json.dumps(chosen,indent=2),flush=True)


if __name__=='__main__':main()
