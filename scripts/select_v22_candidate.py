"""Compare disclosed residual, neural-replacement and direct-blend candidates."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['experiments','features','v20-evidence','v21-evidence','v21-head','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);al=dict(np.load(a.features/'alignment.npz'));old=dict(np.load(a.features/'baseline_heads.npz'))
    ev20=dict(np.load(a.v20_evidence));ev21=dict(np.load(a.v21_evidence));y=ev20['labels'];r=(al['matched_position'][:,None]*10+np.arange(10)).ravel();long=np.repeat(al['long_gap'],10)
    predictions={'first_neural_ce':np.load(a.v21_head)['prediction']}
    for name in ['residual_event3','residual_binary3','residual_event7','ce_seed20261108','ce_seed20261109','gap_weighted']:
        predictions[name]=np.load(a.experiments/name/'predictions.npz')['prediction']
    seed_members=['first_neural_ce','ce_seed20261108','ce_seed20261109'];residual_members=['residual_event3','residual_binary3','residual_event7']
    q=(7*old['tree']+8*old['neural'])/15;small21=q+(4/15)*(predictions['first_neural_ce']-old['neural']);calibration_weight=15/64
    masks={'combined':np.ones(len(y),bool),'earlier':ev20['period']==0,'later':ev20['period']==1}
    baseline_auc={version:{label:float(roc_auc_score(y[m],ev['candidate'][m])) for label,m in masks.items()} for version,ev in [('v20',ev20),('v21',ev21)]}
    gap_auc={version:float(roc_auc_score(al['labels'][long],small[long])) for version,small in [('v20',q),('v21',small21)]};grid=[]
    def evaluate(mode,parent,members,weight):
        new=np.mean([predictions[k] for k in members],axis=0);base=ev20['candidate'] if parent=='v20' else ev21['candidate'];small=q if parent=='v20' else small21
        if mode=='residual_replacement':candidate=base+weight*(new[r]-q[r]);diagnostic=small+(weight/calibration_weight)*(new-q)
        elif mode=='neural_replacement':
            assert parent=='v20';candidate=base+weight*(new[r]-old['neural'][r]);diagnostic=small+(weight/calibration_weight)*(new-old['neural'])
        else:
            candidate=(1-weight)*base+weight*new[r];den=calibration_weight*(1-weight)+weight;diagnostic=(calibration_weight*(1-weight)*small+weight*new)/den
        assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1 and candidate.reshape(-1,10).sum(1).max()<=1+1e-10
        auc={k:float(roc_auc_score(y[m],candidate[m])) for k,m in masks.items()};g20={k:auc[k]-baseline_auc['v20'][k] for k in masks};g21={k:auc[k]-baseline_auc['v21'][k] for k in masks};gap=float(roc_auc_score(al['labels'][long],diagnostic[long])-gap_auc[parent])
        entry={'mode':mode,'parent':parent,'members':list(members),'weight':weight,'auc':auc,'gain_vs_v20':g20,'gain_vs_v21':g21,'long_gap_calibration_gain_vs_parent':gap,'eligible':all(g20[k]>0 and g21[k]>0 for k in ['earlier','later']) and gap>0,'new_models':sum(k!='first_neural_ce' for k in members)}
        grid.append(entry);return candidate
    for n in [1,2,3]:
        for members in itertools.combinations(residual_members,n):
            for parent,weights in [('v20',[calibration_weight/4,calibration_weight/2,calibration_weight]),('v21',[calibration_weight/8,calibration_weight/4,calibration_weight/2])]:
                for w in weights:evaluate('residual_replacement',parent,members,w)
    for members in [seed_members,['gap_weighted']]:
        for w in [.03125,.0625,.125]:evaluate('neural_replacement','v20',members,w)
    # Adaptive final check: adding a head to the full ensemble instead of only replacing a branch.
    for members in [['first_neural_ce'],seed_members,['residual_binary3'],residual_members]:
        for parent in ['v20','v21']:
            for w in [.03125,.0625,.125]:evaluate('direct_blend',parent,members,w)
    (a.out/'grid.json').write_text(json.dumps(grid,indent=2));eligible=[x for x in grid if x['eligible']]
    if not eligible:raise ValueError('No V22 candidate passes the declared comparative checks.')
    best=max(x['gain_vs_v20']['combined'] for x in eligible);selected=min([x for x in eligible if x['gain_vs_v20']['combined']>=best-.000002],key=lambda x:(x['new_models'],len(x['members']),x['weight'],-x['gain_vs_v20']['combined']))
    candidate=evaluate(selected['mode'],selected['parent'],selected['members'],selected['weight'])
    np.savez_compressed(a.out/'evidence.npz',baseline=ev20['candidate'],candidate=candidate,labels=y,users=ev20['users'],period=ev20['period'],indices=ev20['indices'])
    np.savez_compressed(a.out/'increment_evidence.npz',baseline=ev21['candidate'],candidate=candidate,labels=y,users=ev21['users'],period=ev21['period'],indices=ev21['indices'])
    selected.update(version='v22',official_incumbent='v20',official_auc=None,pending_v21_score=None,grid_count=len(grid)-1,selection_tolerance=.000002,
        selection_rule='Both original dates improve over V20 and unscored V21; positive long-gap calibration diagnostic. Within 0.000002 of best eligible gain, prefer fewer additional models, then fewer total heads, then smaller mixture weight.',
        authorization='Standing user instruction permits an exploratory attempt below the unchanged normal gate.',
        limitations='Adaptive model and mixture exploration on reused dates; no independent confirmation. Long-gap diagnostic reuses outcomes and compares calibration subensembles, not the full incumbent.')
    (a.out/'selection.json').write_text(json.dumps(selected,indent=2));print(json.dumps(selected,indent=2),flush=True)
    print('Largest combined-gain candidates:',json.dumps(sorted(grid[:-1],key=lambda x:x['gain_vs_v20']['combined'],reverse=True)[:5],indent=2),flush=True)


if __name__=='__main__':main()
