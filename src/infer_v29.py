"""Run a checksum-locked convex head ensemble on the scored V20 parent."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from common import load_data
from infer_v20 import predict as parent_predict
from calibration_model import predict as head_predict
from v25_exposure_memory import features as memory_features
from validate_submission import validate

PARENT_SHA='3e82021c98bec56bce493b3a56a335b8778e124388293ae5035496bd358c01c5'

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','spec','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--cache',type=Path);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    spec=json.loads(a.spec.read_text())
    def runtime_path(version):
        original=a.runtime/('P3_Labs_v11_model_and_code' if version==11 else f'P3_Labs_v{version}_models')
        compact=a.runtime/f'v{version}'
        if original.is_dir():return original
        if compact.is_dir():return compact
        raise FileNotFoundError(f'Missing runtime v{version}')
    extensions={v:runtime_path(v) for v in [11,12,13,15,16,17,18,19,20]};root=extensions[11]
    if spec['required_v20_csv_sha256']!=PARENT_SHA:raise ValueError('Unknown parent prediction.')
    for filename,hash_ in spec['data_sha256'].items():
        if sha(root/'data'/filename)!=hash_:raise ValueError('Dataset differs from release.')
    if a.cache:
        for filename,hash_ in spec['cache_sha256'].items():
            if sha(a.cache/filename)!=hash_:raise ValueError('Saved inference cache failed its checksum.')
        raw=np.load(a.cache/'raw.npy',mmap_mode='r');names=json.loads((a.cache/'names.json').read_text())
        parts=dict(np.load(a.cache/'components.npz'));ids=parts['rowid'];base=parts['v20']
    else:
        args=[root]+[extensions[v] for v in [12,13,15,16,17,18,19,20]]
        ids,base,raw,names,parts=parent_predict(*args,threads=a.threads,return_components=True)
    serialized=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g')
    if hashlib.sha256(serialized.encode()).hexdigest()!=PARENT_SHA:raise ValueError('Scored V20 parent did not reproduce.')
    reference=np.load(root/'data/evaluation_rowids.npy')
    if not np.array_equal(ids,reference):raise ValueError('Parent row order differs from evaluation.')
    features={'base':(raw,names)}
    if any(head['feature_set']=='memory' for head in spec['heads']):
        fit,query,_=load_data(root/'data');query['target']=np.full(len(query['user']),-1)
        factor=extensions[17]/'models/factor_users_only.pt'
        if sha(factor)!=spec['memory_factor_sha256']:raise ValueError('Memory factor differs from release.')
        ck=torch.load(factor,map_location='cpu',weights_only=False)
        if ck['cut']!=1737936000:raise ValueError('Full factor cutoff differs.')
        extra,en=memory_features(fit,query,ck['state'],1737936000)
        features['memory']=(np.c_[raw,extra],names+en)
    weights=[float(spec['v20_weight'])]+[float(h['weight']) for h in spec['heads']]
    if not np.isfinite(weights).all() or min(weights)<0 or abs(sum(weights)-1)>1e-12:raise ValueError('Nonconvex weights.')
    candidate=weights[0]*base;components={'rowid':ids,'v20':base}
    for entry in spec['heads']:
        path=(a.repo/entry['path']).resolve()
        if not path.is_relative_to(a.repo.resolve()):raise ValueError('Head path escapes repository.')
        if sha(path)!=entry['sha256']:raise ValueError('Head checksum mismatch.')
        head=torch.load(path,map_location='cpu',weights_only=False);x,schema=features[entry['feature_set']]
        if head['names']!=schema:raise ValueError('Feature schema mismatch.')
        prediction=head_predict(head,x);components[entry['name']]=prediction
        candidate=candidate+float(entry['weight'])*prediction
    if not np.isfinite(candidate).all() or candidate.min()<0 or candidate.max()>1 or candidate.reshape(-1,10).sum(1).max()>1+1e-8:
        raise ValueError('Invalid event probabilities.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'rowid':ids,'predicted_probability':candidate}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,reference);mass=pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1)
    if not report['reference_order_matches'] or mass.max()>1+1e-8:raise ValueError('Submission contract failed.')
    report.update(version=spec['version'],parent_csv_sha256=PARENT_SHA,max_event_probability=float(mass.max()),
                  csv_sha256=sha(a.out),spec_sha256=sha(a.spec),source_sha256=sha(Path(__file__)),official_score=None,
                  portal_uploads=0,inference_mode='Verified saved V20 cache' if a.cache else 'Fresh parent inference')
    components['candidate']=candidate;np.savez_compressed(a.out.with_suffix('.components.npz'),**components)
    a.out.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
