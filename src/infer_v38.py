"""Reproduce scored parents and generate the fixed V38 tree/slate exploration."""
import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from common import load_data
from infer_v20 import predict as parent_predict
from calibration_model import predict as head_predict
from v25_exposure_memory import features as memory_features
from v30_tree import predict as tree_predict
from validate_submission import validate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def canonical_sha(path):
    return hashlib.sha256(Path(path).read_bytes().rstrip(b'\n')+b'\n').hexdigest()


def serialize(ids, values):
    return pd.DataFrame({'rowid':ids,'predicted_probability':values}).to_csv(index=False,float_format='%.10g')


def convex_rebuild(parts):
    return (.43963623046875*parts['v17']+.07177734375*parts['neural32']+
            .06280517578125*parts['v38_tree']+.095703125*parts['v27_head']+
            .095703125*parts['v29_head']+.109375*parts['v33_head']+.125*parts['v38_slate'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','spec','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--cache',type=Path);p.add_argument('--threads',type=int,default=3)
    a=p.parse_args();spec=json.loads(a.spec.read_text())
    assert spec['version']=='v38' and spec['recipe']=='tree_plus_slate'
    assert spec['tree_replacement_weight']==.07177734375 and spec['slate_weight']==.125
    assert sha(Path(__file__))==spec['source_sha256']['src/infer_v38.py']
    for path,digest in spec['dependency_canonical_sha256'].items():
        if canonical_sha(a.repo/path)!=digest:raise ValueError('Dependency source changed: '+path)
    parent_spec=a.repo/'artifacts/v33/spec.json'
    if sha(parent_spec)!=spec['required_v33_spec_sha256']:raise ValueError('V33 specification changed.')
    old=json.loads(parent_spec.read_text())
    def runtime_path(version):
        original=a.runtime/('P3_Labs_v11_model_and_code' if version==11 else f'P3_Labs_v{version}_models')
        compact=a.runtime/f'v{version}'
        if original.is_dir():return original
        if compact.is_dir():return compact
        raise FileNotFoundError(f'Missing runtime v{version}')
    extensions={v:runtime_path(v) for v in [11,12,13,15,16,17,18,19,20]};root=extensions[11]
    for filename,digest in old['data_sha256'].items():
        if sha(root/'data'/filename)!=digest:raise ValueError('Dataset checksum mismatch: '+filename)
    if a.cache:
        for filename,digest in old['cache_sha256'].items():
            if sha(a.cache/filename)!=digest:raise ValueError('Parent cache checksum mismatch: '+filename)
        raw=np.load(a.cache/'raw.npy',mmap_mode='r');names=json.loads((a.cache/'names.json').read_text())
        cached=dict(np.load(a.cache/'components.npz'));ids=cached['rowid'];base=cached['v20']
    else:
        args=[root]+[extensions[v] for v in [12,13,15,16,17,18,19,20]]
        ids,base,raw,names,cached=parent_predict(*args,threads=a.threads,return_components=True)
    reference=np.load(root/'data/evaluation_rowids.npy')
    assert np.array_equal(ids,reference) and len(reference)==529390
    parent_hashes={}
    def verify_parent(name,prediction):
        digest=hashlib.sha256(serialize(ids,prediction).encode()).hexdigest()
        expected=spec['required_v33_csv_sha256'] if name=='v33' else old['required_'+name+'_csv_sha256']
        if digest!=expected:raise ValueError('Scored parent did not reproduce: '+name)
        parent_hashes[name]=digest
    verify_parent('v20',base)
    components={'rowid':ids,'v20':base,'v17':cached['v17'],
                'expanded700':cached['expanded700'],'neural32':cached['neural32']}
    v20_rebuilt=.765625*components['v17']+.109375*components['expanded700']+.125*components['neural32']
    assert np.max(abs(v20_rebuilt-base))<5e-16
    fit,query,_=load_data(root/'data');query['target']=np.full(len(query['user']),-1)
    factor=extensions[17]/'models/factor_users_only.pt'
    if sha(factor)!=old['memory_factor_sha256']:raise ValueError('Memory factor differs.')
    ck=torch.load(factor,map_location='cpu',weights_only=False);assert ck['cut']==1737936000
    extra,en=memory_features(fit,query,ck['state'],1737936000)
    memory=np.c_[raw,extra];memory_names=names+en
    def checked_path(entry):
        path=(a.repo/entry['path']).resolve()
        if not path.is_relative_to(a.repo.resolve()):raise ValueError('Model path escapes repository.')
        if sha(path)!=entry['sha256']:raise ValueError('Model checksum mismatch: '+entry['path'])
        return path
    for entry in old['heads']:
        model=torch.load(checked_path(entry),map_location='cpu',weights_only=False)
        x,schema=(raw,names) if entry['feature_set']=='base' else (memory,memory_names)
        assert model['names']==schema
        components[entry['name']]=head_predict(model,x)
    verify_parent('v27',.75*base+.25*components['v27_head'])
    verify_parent('v29',.75*base+.25*components['v29_head'])
    verify_parent('v31',.75*base+.125*components['v27_head']+.125*components['v29_head'])
    incumbent=.65625*base+.109375*components['v27_head']+.109375*components['v29_head']+.125*components['v33_head']
    verify_parent('v33',incumbent);components['v33']=incumbent
    tree=lgb.Booster(model_file=str(checked_path(spec['tree'])))
    tree_spec=json.loads(checked_path(spec['tree_training_spec']).read_text())
    assert tree_spec['names']==names and len(tree_spec['selected'])==269
    tree_parts=[]
    for start in range(0,len(raw),20000):tree_parts.append(tree_predict(tree,raw[start:start+20000],tree_spec))
    components['v38_tree']=np.concatenate(tree_parts)
    slate=torch.load(checked_path(spec['slate_head']),map_location='cpu',weights_only=False)
    assert slate['names']==memory_names and slate['architecture']=='set'
    components['v38_slate']=head_predict(slate,memory)
    candidate=.875*(incumbent+.07177734375*(components['v38_tree']-components['expanded700']))+.125*components['v38_slate']
    components['candidate']=candidate
    convex=convex_rebuild(components);error=float(np.max(abs(candidate-convex)))
    assert error<5e-16
    assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1
    assert candidate.reshape(-1,10).sum(1).max()<=1+1e-8
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p3_v38_',dir=a.out.parent) as temporary:
        staged=Path(temporary)/a.out.name;staged.write_text(serialize(ids,candidate))
        component_path=staged.with_suffix('.components.npz');np.savez_compressed(component_path,**components)
        with zipfile.ZipFile(component_path) as archive:assert archive.testzip() is None
        report=validate(staged,reference)
        assert report['reference_order_matches']
        rounded=pd.read_csv(staged).predicted_probability.to_numpy();max_mass=float(rounded.reshape(-1,10).sum(1).max())
        assert max_mass<=1+1e-8
        report.update(version='v38',spec_sha256=sha(a.spec),source_sha256=sha(Path(__file__)),
          csv_sha256=sha(staged),scored_parent_hashes=parent_hashes,convex_max_abs_error=error,
          max_event_probability=max_mass,components_crc_valid=True,atomic_staged_inference=True,
          release_mode=spec['release_mode'],normal_quality_gate=False,official_score=None,portal_uploads=0,
          inference_mode='Verified saved V20 cache' if a.cache else 'Fresh parent inference')
        staged.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
        for source,target in [(staged,a.out),(component_path,a.out.with_suffix('.components.npz')),
                              (staged.with_suffix('.validation.json'),a.out.with_suffix('.validation.json'))]:
            digest=sha(source)
            if target.exists() and sha(target)!=digest:raise ValueError('Refusing to overwrite different output: '+target.name)
            os.replace(source,target);assert sha(target)==digest
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
