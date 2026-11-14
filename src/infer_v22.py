"""Reproduce V22 using verified cached V20 features and immutable model parents."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np,pandas as pd,lightgbm as lgb,torch
from infer_v21 import compose as compose_v21
from infer_v20 import predict as predict_v20
from infer_v13 import verify_hash
from calibration_residual import features
from flexible_tree import predict as tree_predict
from validate_submission import validate


def csv_bytes(ids,prediction):
    return pd.DataFrame({'rowid':ids,'predicted_probability':prediction}).to_csv(index=False,float_format='%.10g').encode()


def predict(runtime,extension,cache=None,threads=3):
    runtime=Path(runtime);extension=Path(extension);spec=json.loads((extension/'models/v22_spec.json').read_text());torch.set_num_threads(threads)
    for version,sha in spec['required_parent_spec_sha256'].items():
        verify_hash(runtime/f'P3_Labs_{version}_models/models/{version}_spec.json',sha)
    parent=json.loads((runtime/'P3_Labs_v21_models/models/v21_spec.json').read_text())
    if json.loads((runtime/'P3_Labs_v20_models/models/v20_spec.json').read_text())!=parent['required_v20_spec']:
        raise ValueError('V21 parent specification mismatch.')
    if cache is not None:
        cache=Path(cache)
        for file,sha in spec['cache_file_sha256'].items():verify_hash(cache/file,sha)
        parts=dict(np.load(cache/'components.npz'));ids=parts.pop('rowid');base=parts.pop('v20')
        raw=np.load(cache/'raw.npy',mmap_mode='r');names=json.loads((cache/'names.json').read_text())
    else:
        args=[runtime/'P3_Labs_v11_model_and_code']+[runtime/f'P3_Labs_v{v}_models' for v in [12,13,15,16,17,18,19,20]]
        ids,base,raw,names,parts=predict_v20(*args,threads=threads,return_components=True)
    if hashlib.sha256(csv_bytes(ids,base)).hexdigest()!=spec['required_v20_csv_sha256']:raise ValueError('V20 reproduction failed.')
    v21,neural_parts=compose_v21(ids,base,raw,names,parts,runtime/'P3_Labs_v21_models')
    if hashlib.sha256(csv_bytes(ids,v21)).hexdigest()!=spec['required_v21_csv_sha256']:raise ValueError('V21 reproduction failed.')
    if spec['mode']!='direct_blend' or spec['parent']!='v21' or len(spec['heads'])!=1:raise ValueError('Unexpected candidate specification.')
    head=spec['heads'][0];path=extension/'models'/head['file'];verify_hash(path,head['sha256'])
    x,fnames=features(raw,names,parts['expanded700'],parts['neural32']);tree_spec=head['training_spec']['training_spec']
    if fnames!=tree_spec['names']:raise ValueError('Residual feature schema mismatch.')
    model=lgb.Booster(model_file=str(path))
    if model.current_iteration()!=tree_spec['config']['rounds']:raise ValueError('Unexpected number of trees.')
    residual=tree_predict(model,x,tree_spec);weight=spec['weight'];prediction=(1-weight)*v21+weight*residual
    if not 0<=weight<=1 or not np.isfinite(prediction).all() or prediction.min()<0 or prediction.max()>1 or prediction.reshape(-1,10).sum(1).max()>1+1e-12:
        raise ValueError('Invalid probability mixture.')
    components={'v20':base,'v21':v21,'v17':parts['v17'],'v20_tree':parts['expanded700'],'v20_neural':parts['neural32'],
                'first_neural_ce':neural_parts['first_neural_ce'],'residual_binary3':residual}
    return ids,prediction,components


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','extension','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--evidence-out',type=Path);p.add_argument('--report',type=Path);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();ids,prediction,components=predict(a.runtime,a.extension,a.cache,a.threads)
    expected=np.load(a.runtime/'P3_Labs_v11_model_and_code/data/evaluation_rowids.npy')
    if not np.array_equal(ids,expected):raise ValueError('Evaluation row IDs or order differ.')
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_bytes(csv_bytes(ids,prediction))
    if a.evidence_out:np.savez_compressed(a.evidence_out,rowid=ids,prediction=prediction,**components)
    report=validate(a.out,expected);rounded=pd.read_csv(a.out).predicted_probability.to_numpy()
    report.update(max_event_probability=float(rounded.reshape(-1,10).sum(1).max()),v20_csv_reproduced=True,v21_csv_reproduced=True,cache_hashes_verified=a.cache is not None)
    if report['rows']!=529390 or not report['reference_order_matches'] or report['max_event_probability']>1+1e-8:raise ValueError('Submission contract failed.')
    if a.report:a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
