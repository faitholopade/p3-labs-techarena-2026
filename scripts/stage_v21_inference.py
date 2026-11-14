"""Checkpoint the unchanged parent inference chain in short, resumable stages."""
import argparse,hashlib,json,os,pickle,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
import infer_v12,infer_v15,infer_v16,infer_v17,infer_v18,infer_v19,infer_v20


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['v12','v15','v16','v17','v18','v19','v20'],required=True);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    a.cache.mkdir(parents=True,exist_ok=True);a.out.mkdir(parents=True,exist_ok=True)
    root=a.runtime/'P3_Labs_v11_model_and_code';ext={v:a.runtime/f'P3_Labs_v{v}_models' for v in [12,13,15,16,17,18,19,20]}
    repo=Path(__file__).resolve().parents[1]
    identity={'v20_spec':sha(ext[20]/'models/v20_spec.json'),'data':{name:sha(root/'data'/name) for name in ['audit.json','train.npz','evaluation.npz','evaluation_rowids.npy']},
              'inference_sources':{f.name:sha(f) for f in sorted((repo/'src').glob('*.py'))},'threads':a.threads}
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    def load(stage):
        path=a.cache/(stage+'.pickle');meta=json.loads((a.cache/(stage+'.json')).read_text())
        if meta['fingerprint']!=fingerprint or sha(path)!=meta['sha256']:raise ValueError('Stale or damaged stage checkpoint: '+stage)
        # Only load locally produced checkpoints with a matching identity and content digest.
        with path.open('rb') as f:result=pickle.load(f)
        print({'loaded_stage':stage,'sha256':meta['sha256']},flush=True);return result
    if a.stage=='v12': result=infer_v12.components(root,ext[12],a.threads)
    elif a.stage=='v15':
        infer_v15.components=lambda *args,**kwargs:load('v12')
        result=infer_v15.predict(root,ext[12],ext[13],ext[15],a.threads,return_components=True)
    elif a.stage=='v16':
        infer_v16.incumbent_predict=lambda *args,**kwargs:load('v15')
        result=infer_v16.predict(root,ext[12],ext[13],ext[15],ext[16],a.threads,return_components=True)
    elif a.stage=='v17':
        infer_v17.incumbent_predict=lambda *args,**kwargs:load('v16')
        result=infer_v17.predict(root,ext[12],ext[13],ext[15],ext[16],ext[17],a.threads,return_components=True)
    elif a.stage=='v18':
        infer_v18.incumbent_predict=lambda *args,**kwargs:load('v17')
        result=infer_v18.predict(root,ext[12],ext[13],ext[15],ext[16],ext[17],ext[18],a.threads,return_components=True)
    elif a.stage=='v19':
        infer_v19.parent_predict=lambda *args,**kwargs:load('v18')
        result=infer_v19.predict(root,ext[12],ext[13],ext[15],ext[16],ext[17],ext[18],ext[19],a.threads,return_components=True)
    else:
        infer_v20.parent_predict=lambda *args,**kwargs:load('v19')
        result=infer_v20.predict(root,ext[12],ext[13],ext[15],ext[16],ext[17],ext[18],ext[19],ext[20],a.threads,return_components=True)
        ids,base,raw,names,parts=result
        serialized=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g');digest=hashlib.sha256(serialized.encode()).hexdigest()
        if digest!='3e82021c98bec56bce493b3a56a335b8778e124388293ae5035496bd358c01c5':raise ValueError('Scored V20 CSV did not reproduce exactly.')
        np.save(a.out/'raw.npy',raw);(a.out/'names.json').write_text(json.dumps(names));np.savez_compressed(a.out/'components.npz',rowid=ids,v20=base,**parts)
        report={'baseline':'v20','csv_sha256':digest,'scored_baseline_reproduced':True,'rows':len(base),'feature_columns':len(names),'method':'Unchanged parent functions executed in seven saved stages; identity and content hashes checked on every resume.','completed_full_reproductions':1,'earlier_monolithic_runs_interrupted':1}
        (a.out/'reproduction.json').write_text(json.dumps(report,indent=2));print(report,flush=True);return
    path=a.cache/(a.stage+'.pickle');temp=path.with_suffix('.partial')
    with temp.open('wb') as f:pickle.dump(result,f,protocol=5)
    os.replace(temp,path);meta={'stage':a.stage,'fingerprint':fingerprint,'sha256':sha(path),'bytes':path.stat().st_size}
    (a.cache/(a.stage+'.json')).write_text(json.dumps(meta,indent=2));print(meta,flush=True)


if __name__=='__main__':main()
