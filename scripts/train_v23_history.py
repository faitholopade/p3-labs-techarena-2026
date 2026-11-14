"""Fit and score a chronological history-only choice model at one cutoff."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from common import load_data,subset
from history_choice import item_state,features,train,predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','factor','query','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);p.add_argument('--threads',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    data,_,_=load_data(a.root/'data');ix=np.flatnonzero((data['timestamp']<a.cut)&(data['target']<10));fit=subset(data,ix);del data
    ck=torch.load(a.factor,map_location='cpu',weights_only=False)
    if int(ck['cut'])!=a.cut:raise ValueError('Factor cutoff mismatch.')
    items=item_state(ck['state']);del ck;gc.collect();n=len(ix);raw=None;off=np.lib.format.open_memmap(a.out/'training_offset.npy',mode='w+',shape=(n,10),dtype=np.float32)
    for st in range(0,n,2048):
        x,b,names=features(subset(fit,np.arange(st,min(st+2048,n))),items)
        if raw is None:raw=np.lib.format.open_memmap(a.out/'training_features.npy',mode='w+',shape=(n,10,len(names)),dtype=np.float16)
        raw[st:st+len(x)]=x;off[st:st+len(x)]=b
    raw.flush();off.flush();print({'prepared_clicked_events':n,'columns':len(names),'cut':a.cut},flush=True)
    model=train(raw,off,fit['target'],names,threads=a.threads);torch.save(model,a.out/'model.pt')
    query=dict(np.load(a.query));assert np.all(query['target']==-1)
    probability=predict(model,query,items,a.threads);assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1)
    np.savez_compressed(a.out/'prediction.npz',conditional=probability)
    with a.factor.open('rb') as f:sha=hashlib.file_digest(f,'sha256').hexdigest()
    report={'cut':a.cut,'factor_file':str(a.factor),'factor_sha256':sha,'training_clicked_events':n,'max_training_timestamp':int(fit['timestamp'].max()),'validation_events':len(query['items']),'columns':len(names),'query_targets_masked':True,'user_factor_inputs':False,'labels_after_cutoff_used_for_fit':False,'curve':model['curve'],'limits':'Item embeddings learned on the same prior training window; current validation dates are weekends. No weekday confirmation.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print({'complete':str(a.out),'query_events':len(query['items'])},flush=True)


if __name__=='__main__':main()
