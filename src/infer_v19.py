"""Reproduce V19 while checking the exact scored V18 parent submission."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch,lightgbm as lgb
from infer_v18 import predict as parent_predict
from infer_v13 import verify_hash
from history_tree_features import extend
from flexible_tree import predict as predict_tree
from validate_submission import validate


def predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,v18_extension,extension,threads=3,evidence_out=None,return_components=False):
    extension=Path(extension);v17_extension=Path(v17_extension);v18_extension=Path(v18_extension)
    spec=json.loads((extension/'models/v19_spec.json').read_text());parent=json.loads((v18_extension/'models/v18_spec.json').read_text())
    if parent!=spec['required_v18_spec']:raise ValueError('Unexpected V18 specification.')
    if parent['weights']!=[.875,.125] or len(parent['tree_models'])!=1:raise ValueError('Unexpected parent composition.')
    ids,base,raw,events,names,parts=parent_predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,v18_extension,threads,return_components=True)
    serialized=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g');sha=hashlib.sha256(serialized.encode()).hexdigest();del serialized
    if sha!=spec['required_v18_csv_sha256']:raise ValueError('Scored V18 CSV did not reproduce.')
    print({'verified_v18_csv_sha256':sha},flush=True)
    factor=v17_extension/'models/factor_users_only.pt';verify_hash(factor,spec['required_feature_factor_sha256'])
    state=torch.load(factor,map_location='cpu',weights_only=False)['state'];raw,names=extend(raw,events,state,names);new=[]
    for entry in spec['tree_models']:
        path=extension/'models'/entry['file'];verify_hash(path,entry['sha256']);ts=entry['training_spec']
        if names!=ts['names']:raise ValueError('Feature schema mismatch.')
        model=lgb.Booster(model_file=str(path));new.append(predict_tree(model,raw,ts));print({'scored_v19_tree':entry['name']},flush=True)
    weights=np.asarray(spec['weights'],float)
    if len(weights)!=2+len(new) or (weights<0).any() or not np.isclose(weights.sum(),1):raise ValueError('Invalid convex weights.')
    pred=weights[0]*parts['v17']+weights[1]*parts['tree_predictions'][0]
    for weight,prob in zip(weights[2:],new):pred=pred+weight*prob
    if not np.isfinite(pred).all() or pred.min()<0 or pred.max()>1 or pred.reshape(-1,10).sum(1).max()>1+1e-12:raise ValueError('Invalid probabilities.')
    if evidence_out is not None:np.savez_compressed(evidence_out,rowid=ids,baseline=base,v17=parts['v17'],old_tree=parts['tree_predictions'][0],new_trees=np.asarray(new),prediction=pred)
    if return_components:return ids,pred,raw,names,{'v17':parts['v17'],'v19_trees':np.asarray(new)}
    return ids,pred


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','v12-extension','v13-extension','v15-extension','v16-extension','v17-extension','v18-extension','extension','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--evidence-out');a=p.parse_args()
    ids,pred=predict(a.root,a.v12_extension,a.v13_extension,a.v15_extension,a.v16_extension,a.v17_extension,a.v18_extension,a.extension,a.threads,a.evidence_out)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'rowid':ids,'predicted_probability':pred}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,ids);report['max_event_probability']=float(pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1).max())
    if not report['reference_order_matches'] or report['max_event_probability']>1+1e-8:raise ValueError('Submission contract failed.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
