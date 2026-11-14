"""Assess fixed history-choice and chronological propensity mixtures against V20."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--mode',choices=['history','event_rate'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    al=np.load(a.features/'alignment.npz');ev=np.load(a.incumbent);base=ev['candidate'].reshape(-1,10);idx=al['matched_position'];old=np.load(a.features/'baseline_heads.npz');small=((7*old['tree']+8*old['neural'])/15).reshape(-1,10);gap=np.repeat(al['long_gap'],10)
    key='conditional' if a.mode=='history' else 'event_rate';values=np.concatenate([np.load(a.experiments/(a.mode+'_'+period)/'prediction.npz')[key] for period in ['earlier','later']])
    if a.mode=='history':
        new=values[idx]*base.sum(1,keepdims=True);diagnostic=values*small.sum(1,keepdims=True);weights=[.03125,.0625,.125]
    else:
        new=base/base.sum(1,keepdims=True)*values[idx,None];diagnostic=small/small.sum(1,keepdims=True)*values[:,None];weights=[.0625,.125,.25]
    results=[]
    for weight in weights:
        prediction=((1-weight)*base+weight*new).ravel();diag=((1-weight)*small+weight*diagnostic).ravel();gain={};auc={}
        for name,m in [('combined',np.ones(len(prediction),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[name]=float(roc_auc_score(ev['labels'][m],prediction[m]));gain[name]=auc[name]-float(roc_auc_score(ev['labels'][m],ev['candidate'][m]))
        g=float(roc_auc_score(al['labels'][gap],diag[gap])-roc_auc_score(al['labels'][gap],small.ravel()[gap]))
        results.append({'weight':weight,'auc':auc,'gain_vs_v20':gain,'long_gap_calibration_gain':g})
    report={'mode':a.mode,'results':results,'standalone_with_v20_complement_auc':float(roc_auc_score(ev['labels'],new.ravel())),'limits':'Both dates are weekends; long-gap comparison is calibration-only and reuses outcomes.'}
    np.savez_compressed(a.out/'predictions.npz',raw_output=values,standalone=new.ravel());(a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
