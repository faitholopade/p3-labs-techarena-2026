"""Standalone V142 production inference; exact saved V136 CSV values as base."""
import argparse,hashlib
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2]
def predict(model,inputs,base):
    c=inputs['items'];ctx=inputs['context'];x=np.clip((model['catalog'][c]-model['mean'])/model['std'],-6,6);x-=x.mean(1,keepdims=True);linear=np.einsum('nkd,d->nk',x,model['weight'][0])*((ctx//4)-2/7)[:,None];d=.2*np.tanh(linear/.2);b=base.reshape(-1,10);z=np.log(b)+.5*d;z-=z.max(1,keepdims=True);weights=np.exp(z);p=weights/weights.sum(1,keepdims=True)*b.sum(1,keepdims=True);return p.ravel(),d
def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();folder=ROOT/'runtime/v142';model=np.load(folder/'production_model.npz');inputs=np.load(folder/'production_inputs.npz');base=np.load(folder/'production_baseline.npz')['probabilities'];values,_=predict(model,inputs,base);a.out.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame({'rowid':inputs['rowids'],'predicted_probability':values}).to_csv(a.out,index=False,float_format='%.10g',lineterminator='\n')
    with a.out.open('rb') as f:print(dict(rows=len(values),sha256=hashlib.file_digest(f,'sha256').hexdigest()))
if __name__=='__main__':main()
