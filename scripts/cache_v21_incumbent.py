"""Reproduce the scored V20 CSV exactly and cache its outcome-free features."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from infer_v20 import predict


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,default=3);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    args=[a.runtime/'P3_Labs_v11_model_and_code']+[a.runtime/f'P3_Labs_v{v}_models' for v in [12,13,15,16,17,18,19,20]]
    ids,base,raw,names,parts=predict(*args,threads=a.threads,return_components=True)
    text=pd.DataFrame({'rowid':ids,'predicted_probability':base}).to_csv(index=False,float_format='%.10g');sha=hashlib.sha256(text.encode()).hexdigest()
    if sha!='3e82021c98bec56bce493b3a56a335b8778e124388293ae5035496bd358c01c5':raise ValueError('Scored V20 did not reproduce exactly.')
    np.save(a.out/'raw.npy',raw);(a.out/'names.json').write_text(json.dumps(names));np.savez_compressed(a.out/'components.npz',rowid=ids,v20=base,**parts)
    report={'baseline':'v20','csv_sha256':sha,'scored_baseline_reproduced':True,'rows':len(base),'feature_columns':len(names)}
    (a.out/'reproduction.json').write_text(json.dumps(report,indent=2));print(report,flush=True)


if __name__=='__main__':main()
