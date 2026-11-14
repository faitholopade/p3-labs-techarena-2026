"""Reproduce the V18 tree ensemble while verifying the scored V17 baseline."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb
from infer_v17 import predict as incumbent_predict
from infer_v13 import verify_hash
from event_tree import predict as tree_predict
from validate_submission import validate


def utility(p):
    p=np.asarray(p,np.float64).reshape(-1,10)
    return np.log(np.maximum(p,1e-9)/np.maximum(1-p.sum(1,keepdims=True),1e-9)).ravel()


def predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,extension,threads=3,evidence_out=None,return_components=False):
    extension=Path(extension);v17_extension=Path(v17_extension)
    spec=json.loads((extension/'models/v18_spec.json').read_text());parent=json.loads((v17_extension/'models/v17_spec.json').read_text())
    if parent!=spec['required_v17_spec']:raise ValueError('Unexpected V17 baseline specification.')
    weights=np.asarray(spec['weights'],float)
    if len(weights)!=1+len(spec['tree_models']) or (weights<0).any() or not np.isclose(weights.sum(),1):raise ValueError('Invalid convex mixture.')
    ids,base,raw,events,names,parts=incumbent_predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,threads,return_components=True)
    base_csv=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g')
    baseline_sha=hashlib.sha256(base_csv.encode()).hexdigest();del base_csv
    if baseline_sha!=spec['required_v17_csv_sha256']:raise ValueError('V17 did not reproduce the scored CSV.')
    # V17 returns features for the last factor it scores, verified here.
    if parent['factor_models'][-1]!='factor_users_only.pt' or parent['factor_posterior_samples'][-1]!=256:raise ValueError('Unexpected parent feature state.')
    offset=utility(parts['factor_users_only.pt']);predictions=[]
    for entry in spec['tree_models']:
        path=extension/'models'/entry['file'];verify_hash(path,entry['sha256']);ts=entry['training_spec']
        if ts['names']!=names or ts['feature_spec']['factor_name']!='factor_users_only.pt' or ts['feature_spec']['posterior_samples']!=256:raise ValueError('Tree feature schema mismatch.')
        model=lgb.Booster(model_file=str(path));prob=tree_predict(model,raw,ts,external_offset=offset if ts.get('external_offset',False) else None)
        predictions.append(prob);print({'scored_v18_tree':entry['file']},flush=True)
    prediction=weights[0]*base
    for w,prob in zip(weights[1:],predictions):prediction=prediction+w*prob
    if not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1 or prediction.reshape(-1,10).sum(1).max()>1+1e-12:raise ValueError('Invalid final probabilities.')
    if evidence_out is not None:np.savez_compressed(evidence_out,rowid=ids,baseline=base,tree_predictions=np.asarray(predictions),prediction=prediction)
    print({'verified_v17_csv_sha256':baseline_sha},flush=True)
    if return_components:return ids,prediction,raw,events,names,{'v17':base,'tree_predictions':np.asarray(predictions),'factor_predictions':parts}
    return ids,prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','v12-extension','v13-extension','v15-extension','v16-extension','v17-extension','extension','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--evidence-out')
    a=p.parse_args();ids,pred=predict(a.root,a.v12_extension,a.v13_extension,a.v15_extension,a.v16_extension,a.v17_extension,a.extension,a.threads,a.evidence_out)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'rowid':ids,'predicted_probability':pred}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,ids);report['max_event_probability']=float(pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1).max())
    if not report['reference_order_matches'] or report['max_event_probability']>1+1e-8:raise ValueError('Rounded submission contract failed.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
