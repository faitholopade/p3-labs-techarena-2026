"""Reproduce the selected V17 factor ensemble against the scored V16 baseline."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infer_v16 import predict as incumbent_predict
from infer_v13 import verify_hash
from factor_ensemble import update_features
from sequence_calibration import history_pairs,predict as sequence_predict
from validate_submission import validate


def predict(root,v12_extension,v13_extension,v15_extension,v16_extension,extension,threads=3,evidence_out=None,return_components=False):
    root,v12_extension,v13_extension,v15_extension,v16_extension,extension=map(Path,[root,v12_extension,v13_extension,v15_extension,v16_extension,extension])
    spec=json.loads((extension/'models/v17_spec.json').read_text())
    old=json.loads((v16_extension/'models/v16_spec.json').read_text())
    if old!=spec['required_v16_spec']:raise ValueError('Unexpected V16 baseline specification.')
    weights=np.asarray(spec['weights'],float);factor_weights=np.asarray(spec['factor_weights'],float)
    if weights.shape!=(4,) or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Expected convex weights for V16, V15, V8, and new sequence predictions.')
    if factor_weights.shape!=(len(spec['factor_models']),) or (factor_weights<0).any() or not np.isclose(factor_weights.sum(),1):
        raise ValueError('Invalid factor weights.')
    samples=spec['factor_posterior_samples']
    if len(samples)!=len(factor_weights) or any(not isinstance(x,int) or x<=0 for x in samples):
        raise ValueError('Invalid posterior sample counts.')
    factors=[]
    for filename in spec['factor_models']:
        path=extension/'models'/filename;verify_hash(path,spec['factor_sha256'][filename]);factors.append(path)
    ids,baseline,v15,legacy,old_sequence,raw,events,names=incumbent_predict(
        root,v12_extension,v13_extension,v15_extension,v16_extension,threads,return_components=True)
    baseline_csv=pd.DataFrame({'rowid':ids,'predicted_probability':baseline}).to_csv(index=False,float_format='%.10g')
    baseline_sha=hashlib.sha256(baseline_csv.encode()).hexdigest();del baseline_csv
    if baseline_sha!=spec['required_v16_csv_sha256']:raise ValueError('V16 baseline did not reproduce the scored CSV.')
    print({'verified_v16_csv_sha256':baseline_sha},flush=True)
    heads=[]
    head_spec=old['required_v15_spec']['required_v13_spec']
    for filename in head_spec['sequence_models']:
        path=v13_extension/'models'/filename;verify_hash(path,head_spec['sequence_model_sha256'][filename])
        head=torch.load(path,map_location='cpu',weights_only=False)
        if head['base_spec']['names']!=names:raise ValueError('Feature schema mismatch.')
        heads.append(head)
    parts=[]
    for path,sample_count in zip(factors,samples):
        update_features(raw,events,path,posterior_samples=sample_count)
        state=torch.load(path,map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,events)
        parts.append(np.mean([sequence_predict(h,raw,pairs,mask) for h in heads],axis=0))
        del state,pairs;gc.collect();print({'scored_v17_factor':path.name},flush=True)
    sequence=np.average(parts,axis=0,weights=factor_weights)
    prediction=weights[0]*baseline+weights[1]*v15+weights[2]*legacy+weights[3]*sequence
    if not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1:
        raise ValueError('Invalid final probabilities.')
    if prediction.reshape(-1,10).sum(1).max()>1+1e-12:raise ValueError('Invalid event probability mass.')
    if evidence_out is not None:
        np.savez_compressed(evidence_out,rowid=ids,baseline=baseline,v15=v15,legacy=legacy,
                            incumbent_sequence=old_sequence,factor_predictions=np.asarray(parts),sequence=sequence,prediction=prediction)
    if return_components:return ids,prediction,raw,events,names,dict(zip(spec['factor_models'],parts))
    return ids,prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['root','v12-extension','v13-extension','v15-extension','v16-extension','extension','out']:p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--evidence-out')
    a=p.parse_args();ids,pred=predict(a.root,a.v12_extension,a.v13_extension,a.v15_extension,a.v16_extension,a.extension,a.threads,a.evidence_out)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'rowid':ids,'predicted_probability':pred}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,ids)
    if not report['reference_order_matches']:raise ValueError('Submission row order mismatch.')
    report['max_event_probability']=float(pd.read_csv(a.out)['predicted_probability'].to_numpy().reshape(-1,10).sum(1).max())
    if report['max_event_probability']>1+1e-8:raise ValueError('Rounded event probability exceeds one.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
