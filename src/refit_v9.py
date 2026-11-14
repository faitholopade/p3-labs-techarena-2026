"""Regenerate chronological features and refit the selected V9 neural heads."""
import argparse,json,gc
from pathlib import Path
import numpy as np,torch
from common import load_data,subset,binary_labels
from daily_inference import build
from daily_training import train

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');ap.add_argument('--cache',default='scratch_v9');ap.add_argument('--out',default='refit_v9_models');a=ap.parse_args();root=Path(a.root);cache=Path(a.cache);cache.mkdir(parents=True,exist_ok=True);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d,_,audit=load_data(root/'data');spec=json.loads((root/'models/v9_spec.json').read_text());ds=[]
    for day,cut,model in [(24,1737676800,root/'validation/v9/source_models/residual_jan24.pt'),(25,1737763200,root/'validation/v8/source_models/residual_earlier.pt'),(26,1737849600,root/'validation/v8/source_models/residual_later.pt')]:
        dest=cache/f'day{day}';dest.mkdir(exist_ok=True);ix=np.flatnonzero((d['timestamp']>=cut)&(d['timestamp']<cut+86400));fit=subset(d,np.flatnonzero(d['timestamp']<cut));query=subset(d,ix);x,names=build(fit,query,cut,audit['n_items'],audit['n_users'],model,state_root=root/'validation/v9/source_states');np.save(dest/'features.npy',x);del x,fit;gc.collect();ds.append({'x':np.load(dest/'features.npy',mmap_mode='r'),'names':np.array(names),'labels':binary_labels(query).ravel(),'user':query['user']});print({'day':day,'events':len(ix)},flush=True)
    for seed,filename in zip(spec['seeds'],spec['models']):
        ck=train(ds,[np.ones(len(z['user']),bool) for z in ds],seed,spec['epochs'],spec['width']);torch.save(ck,out/filename);print({'saved_model':filename},flush=True)
if __name__=='__main__':main()
