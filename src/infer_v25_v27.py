"""Apply a specified V25–V27 head mixture to the verified scored V20 parent."""
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
    for key in ['runtime','models','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cache',type=Path);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    spec=json.loads((a.models/'spec.json').read_text())
    def runtime_path(version):
        original=a.runtime/('P3_Labs_v11_model_and_code' if version==11 else f'P3_Labs_v{version}_models')
        compact=a.runtime/f'v{version}'
        if original.is_dir():return original
        if compact.is_dir():return compact
        raise FileNotFoundError(f'Missing runtime v{version}: expected {original} or {compact}')
    extensions={v:runtime_path(v) for v in [11,12,13,15,16,17,18,19,20]};root=extensions[11]
    if spec['required_v20_csv_sha256']!=PARENT_SHA:raise ValueError('Unknown parent prediction.')
    for filename,hash_ in spec['data_sha256'].items():
        if sha(root/'data'/filename)!=hash_:raise ValueError('Dataset differs from release.')
    v20spec=json.loads((extensions[20]/'models/v20_spec.json').read_text())
    old_name=next(e['name'] for e in v20spec['heads'] if e['kind']=='neural')
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
    if spec['family']=='v25':
        fit,query,_=load_data(root/'data');query['target']=np.full(len(query['user']),-1)
        factor=extensions[17]/'models/factor_users_only.pt'
        if sha(factor)!=spec['memory_factor_sha256']:raise ValueError('Memory factor differs from release.')
        ck=torch.load(factor,map_location='cpu',weights_only=False)
        if ck['cut']!=1737936000:raise ValueError('Full factor cutoff differs.')
        extra,en=memory_features(fit,query,ck['state'],1737936000);raw=np.c_[raw,extra];names=names+en
    predictions=[]
    for entry in spec['heads']:
        path=a.models/entry['file']
        if sha(path)!=entry['sha256']:raise ValueError('Head checksum mismatch.')
        head=torch.load(path,map_location='cpu',weights_only=False)
        if head['names']!=names:raise ValueError('Feature schema mismatch.')
        predictions.append(head_predict(head,raw))
    if spec.get('include_old_neural',False):predictions.append(parts[old_name])
    head_mean=np.mean(predictions,axis=0);weight=float(spec['weight'])
    if not 0<=weight<=1:raise ValueError('Invalid mixture weight.')
    if spec['mode']=='blend':candidate=(1-weight)*base+weight*head_mean
    elif spec['mode']=='replace_neural':
        if weight!=.125:raise ValueError('Only the original neural weight can be replaced.')
        candidate=base+weight*(head_mean-parts[old_name])
    else:raise ValueError('Unknown mixture mode.')
    if not np.isfinite(candidate).all() or np.any(candidate<0) or np.any(candidate>1) or candidate.reshape(-1,10).sum(1).max()>1+1e-7:
        raise ValueError('Invalid event probabilities.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({'rowid':ids,'predicted_probability':candidate}).to_csv(a.out,index=False,float_format='%.10g')
    report=validate(a.out,reference)
    mass=pd.read_csv(a.out).predicted_probability.to_numpy().reshape(-1,10).sum(1)
    if not report['reference_order_matches'] or mass.max()>1+1e-8:raise ValueError('Submission contract failed.')
    report.update({'version':spec['family'],'parent_csv_sha256':PARENT_SHA,'max_event_probability':float(mass.max()),
                   'csv_sha256':sha(a.out),'source_sha256':sha(Path(__file__)),'official_score':None,'portal_uploads':0,
                   'inference_mode':'Verified saved V20 cache' if a.cache else 'Fresh parent inference'})
    np.savez_compressed(a.out.with_suffix('.components.npz'),rowid=ids,baseline=base,head_mean=head_mean,old_neural=parts[old_name],candidate=candidate)
    a.out.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
