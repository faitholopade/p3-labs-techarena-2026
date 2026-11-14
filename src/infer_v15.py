"""Reproduce the V15 CTR candidate from the V11/V12/V13 runtimes and V15 factors."""
import argparse,gc,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from infer_v12 import components
from infer_v13 import verify_hash
from factor_ensemble import update_features
from sequence_calibration import history_pairs,predict as predict_sequence
from validate_submission import validate


def predict(root,v12_extension,v13_extension,extension,threads=3,evidence_out=None,return_components=False):
    root,v12_extension,v13_extension,extension=map(Path,[root,v12_extension,v13_extension,extension])
    spec=json.loads((extension/'models/v15_spec.json').read_text())
    old=json.loads((v13_extension/'models/v13_spec.json').read_text())
    if old!=spec['required_v13_spec']:
        raise ValueError('The V13 specification differs from the required baseline.')
    if json.loads((v12_extension/'models/v12_spec.json').read_text())!=old['required_v12_spec']:
        raise ValueError('The V12 specification differs from the required baseline.')
    weights=np.asarray(spec['weights'],float);factor_weights=np.asarray(spec['factor_weights'],float)
    if weights.shape!=(3,) or (weights<0).any() or not np.isclose(weights.sum(),1):
        raise ValueError('Expected three nonnegative component weights summing to one.')
    if factor_weights.shape!=(len(spec['factor_models']),) or (factor_weights<0).any() or not np.isclose(factor_weights.sum(),1):
        raise ValueError('Invalid factor-ensemble weights.')
    factors=[]
    for filename in spec['factor_models']:
        path=extension/'models'/filename;verify_hash(path,spec['factor_sha256'][filename]);factors.append(path)
    samples=spec.get('factor_posterior_samples',[spec['posterior_samples']]*len(factors))
    if len(samples)!=len(factors) or any(not isinstance(n,int) or n<=0 for n in samples):
        raise ValueError('Invalid per-factor posterior sample counts.')
    heads=[]
    for filename in old['sequence_models']:
        path=v13_extension/'models'/filename;verify_hash(path,old['sequence_model_sha256'][filename])
        heads.append(torch.load(path,map_location='cpu',weights_only=False))
    ids,v12,legacy,raw,events,names=components(root,v12_extension,threads)
    if any(head['base_spec']['names']!=names for head in heads):
        raise ValueError('Sequence-head feature schema mismatch.')
    predictions=[]
    for factor,sample_count in zip(factors,samples):
        update_features(raw,events,factor,posterior_samples=sample_count)
        state=torch.load(factor,map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,events)
        predictions.append(np.mean([predict_sequence(head,raw,pairs,mask) for head in heads],axis=0))
        del state,pairs;gc.collect();print({'scored_factor':factor.name},flush=True)
    sequence=np.average(predictions,axis=0,weights=factor_weights)
    probability=weights[0]*v12+weights[1]*legacy+weights[2]*sequence
    if not np.isfinite(probability).all() or probability.min()<0 or probability.max()>1:
        raise ValueError('Invalid final probabilities.')
    if probability.reshape(-1,10).sum(1).max()>1+1e-12:
        raise ValueError('Event click probability exceeds one.')
    if evidence_out is not None:
        np.savez_compressed(evidence_out,rowid=ids,v12=v12,legacy=legacy,sequence=sequence,
                            factor_predictions=np.asarray(predictions),prediction=probability)
    if return_components:return ids,probability,v12,legacy,raw,events,names
    return ids,probability


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',required=True)
    p.add_argument('--v12-extension',required=True);p.add_argument('--v13-extension',required=True)
    p.add_argument('--extension',required=True);p.add_argument('--out',required=True)
    p.add_argument('--evidence-out');p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();ids,prediction=predict(a.root,a.v12_extension,a.v13_extension,a.extension,a.threads,a.evidence_out)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'rowid':ids,'predicted_probability':prediction}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,ids)
    if not report['reference_order_matches']:raise ValueError('Submission row order differs from the example.')
    rounded=pd.read_csv(a.out)['predicted_probability'].to_numpy().reshape(-1,10)
    report['max_event_probability']=float(rounded.sum(1).max())
    if report['max_event_probability']>1+1e-8:raise ValueError('Rounded event probability exceeds one.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
