"""Reproduce the V21 first-event calibration mixture from its immutable parents."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,torch
from infer_v20 import predict as parent_predict
from infer_v13 import verify_hash
from calibration_model import predict as neural_predict
from validate_submission import validate


def compose(ids,base,raw,names,parts,extension):
    extension=Path(extension);spec=json.loads((extension/'models/v21_spec.json').read_text())
    serialized=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g')
    sha=hashlib.sha256(serialized.encode()).hexdigest()
    if sha!=spec['required_v20_csv_sha256']:raise ValueError('Scored V20 CSV did not reproduce.')
    components={'v20':base,'v20_neural':parts['neural32']};fresh=[]
    for entry in spec['heads']:
        path=extension/'models'/entry['file'];verify_hash(path,entry['sha256'])
        model=torch.load(path,map_location='cpu',weights_only=False)
        if names!=model['names']:raise ValueError('Feature schema mismatch.')
        pred=neural_predict(model,raw);components[entry['name']]=pred;fresh.append(pred)
    prediction=base+spec['replacement_weight']*(np.mean(fresh,axis=0)-parts['neural32'])
    if not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1 or prediction.reshape(-1,10).sum(1).max()>1+1e-12:raise ValueError('Invalid probabilities.')
    return prediction,components


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runtime',type=Path,required=True);p.add_argument('--extension',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--cache',type=Path);p.add_argument('--evidence-out',type=Path);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();spec=json.loads((a.extension/'models/v21_spec.json').read_text())
    parent=json.loads((a.runtime/'P3_Labs_v20_models/models/v20_spec.json').read_text())
    if parent!=spec['required_v20_spec']:raise ValueError('Unexpected V20 model specification.')
    if a.cache:
        parts=dict(np.load(a.cache/'components.npz'));ids=parts.pop('rowid');base=parts.pop('v20');raw=np.load(a.cache/'raw.npy',mmap_mode='r');names=json.loads((a.cache/'names.json').read_text())
    else:
        args=[a.runtime/'P3_Labs_v11_model_and_code']+[a.runtime/f'P3_Labs_v{v}_models' for v in [12,13,15,16,17,18,19,20]]
        ids,base,raw,names,parts=parent_predict(*args,threads=a.threads,return_components=True)
    prediction,components=compose(ids,base,raw,names,parts,a.extension)
    a.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'rowid':ids,'predicted_probability':prediction}).to_csv(a.out,index=False,float_format='%.10g')
    if a.evidence_out:np.savez_compressed(a.evidence_out,rowid=ids,prediction=prediction,**components)
    report=validate(a.out,ids);report['max_event_probability']=float(pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1).max())
    if report['rows']!=529390 or not report['reference_order_matches'] or report['max_event_probability']>1+1e-8:raise ValueError('Submission contract failed.')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
