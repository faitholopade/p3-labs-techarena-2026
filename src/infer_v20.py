"""Reproduce expanded-data V20 heads and verify the exact scored V19 parent."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch,lightgbm as lgb
from infer_v19 import predict as parent_predict
from infer_v13 import verify_hash
from flexible_tree import predict as tree_predict
from calibration_model import predict as neural_predict
from validate_submission import validate


def predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,v18_extension,v19_extension,extension,threads=3,evidence_out=None,return_components=False):
    extension=Path(extension);v19_extension=Path(v19_extension)
    spec=json.loads((extension/'models/v20_spec.json').read_text());parent=json.loads((v19_extension/'models/v19_spec.json').read_text())
    if parent!=spec['required_v19_spec']:raise ValueError('Unexpected V19 parent specification.')
    ids,baseline,raw,names,parts=parent_predict(root,v12_extension,v13_extension,v15_extension,v16_extension,v17_extension,v18_extension,v19_extension,threads,return_components=True)
    serialized=pd.DataFrame({'rowid':ids,'predicted_probability':baseline}).to_csv(index=False,float_format='%.10g');sha=hashlib.sha256(serialized.encode()).hexdigest();del serialized
    if sha!=spec['required_v19_csv_sha256']:raise ValueError('Scored V19 CSV did not reproduce.')
    print({'verified_v19_csv_sha256':sha},flush=True);components={'v19':baseline,'v17':parts['v17']}
    for entry in spec['heads']:
        path=extension/'models'/entry['file'];verify_hash(path,entry['sha256'])
        if entry['kind']=='tree':
            hs=entry['training_spec'];assert names==hs['names'];model=lgb.Booster(model_file=str(path));p=tree_predict(model,raw,hs)
        elif entry['kind']=='neural':
            model=torch.load(path,map_location='cpu',weights_only=False);assert names==model['names'];p=neural_predict(model,raw)
        else:raise ValueError('Unsupported head kind.')
        components[entry['name']]=p;print({'scored_v20_head':entry['name']},flush=True)
    weights=np.asarray(spec['weights'],float);order=spec['weight_order']
    if len(weights)!=len(order) or np.any(weights<0) or not np.isclose(weights.sum(),1):raise ValueError('Invalid convex weights.')
    prediction=sum(float(w)*components[k] for k,w in zip(order,weights) if w>0)
    if not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1 or prediction.reshape(-1,10).sum(1).max()>1+1e-12:raise ValueError('Invalid probabilities.')
    if evidence_out is not None:np.savez_compressed(evidence_out,rowid=ids,prediction=prediction,**components)
    if return_components:return ids,prediction,raw,names,components
    return ids,prediction


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','v12-extension','v13-extension','v15-extension','v16-extension','v17-extension','v18-extension','v19-extension','extension','out']:p.add_argument('--'+key,required=True)
    p.add_argument('--threads',type=int,default=3);p.add_argument('--evidence-out');a=p.parse_args()
    ids,pred=predict(a.root,a.v12_extension,a.v13_extension,a.v15_extension,a.v16_extension,a.v17_extension,a.v18_extension,a.v19_extension,a.extension,a.threads,a.evidence_out)
    Path(a.out).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'rowid':ids,'predicted_probability':pred}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,ids);report['max_event_probability']=float(pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1).max())
    if not report['reference_order_matches'] or report['max_event_probability']>1+1e-8:raise ValueError('Submission contract failed.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
