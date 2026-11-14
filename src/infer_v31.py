"""Stage V31 inference, verify both scored parents, then publish complete files."""
import argparse,errno,hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','spec','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repo',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--cache',type=Path);p.add_argument('--threads',type=int,default=3);a=p.parse_args()
    spec=json.loads(a.spec.read_text());assert spec['version']=='v31'
    if spec['v20_weight']!=.75 or [(h['name'],h['weight']) for h in spec['heads']]!=[('v27_head',.125),('v29_head',.125)]:raise ValueError('Unexpected midpoint recipe.')
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='p3_v31_infer_',dir='/tmp') as work:
        staged=Path(work)/a.out.name
        args=[sys.executable,str((a.repo/'src/infer_v29.py').resolve()),'--runtime',str(a.runtime.resolve()),'--spec',str(a.spec.resolve()),'--repo',str(a.repo.resolve()),'--out',str(staged),'--threads',str(a.threads)]
        if a.cache:args+=['--cache',str(a.cache.resolve())]
        subprocess.run(args,check=True)
        component_path=staged.with_suffix('.components.npz')
        with zipfile.ZipFile(component_path) as z:
            if z.testzip() is not None:raise ValueError('Invalid component archive.')
        parts=dict(np.load(component_path));parent_hashes={}
        for name,key in [('v27','v27_head'),('v29','v29_head')]:
            parent=.75*parts['v20']+.25*parts[key]
            serialized=pd.DataFrame({'rowid':parts['rowid'],'predicted_probability':parent}).to_csv(index=False,float_format='%.10g')
            digest=hashlib.sha256(serialized.encode()).hexdigest()
            if digest!=spec['required_'+name+'_csv_sha256']:raise ValueError(f'{name} scored parent did not reproduce.')
            parent_hashes[name]=digest
        report=json.loads(staged.with_suffix('.validation.json').read_text());report.update(scored_parent_hashes=parent_hashes,components_crc_valid=True,wrapper_sha256=sha(Path(__file__)),atomic_staged_inference=True)
        staged.with_suffix('.validation.json').write_text(json.dumps(report,indent=2)+'\n')
        for source,target in [(staged,a.out),(component_path,a.out.with_suffix('.components.npz')),(staged.with_suffix('.validation.json'),a.out.with_suffix('.validation.json'))]:
            expected=sha(source)
            if target.exists() and sha(target)!=expected:raise ValueError(f'Refusing to replace a different existing output: {target.name}')
            try:os.replace(source,target)
            except OSError as error:
                if error.errno!=errno.EXDEV:raise
                tmp=target.with_suffix(target.suffix+'.partial');shutil.copyfile(source,tmp)
                if sha(tmp)!=expected:raise ValueError('Staged transfer checksum mismatch.')
                os.replace(tmp,target)
            if sha(target)!=expected:raise ValueError('Published output checksum mismatch.')
        print(json.dumps(report),flush=True)

if __name__=='__main__':main()
