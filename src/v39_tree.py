"""Pooled memory event tree with bounded temporary storage and explicit offset."""
import gc
import shutil
import tempfile
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
from flexible_tree import probability
from v30_features import stable_columns


def fit(datasets, masks, config):
    names=datasets[0]['names'].tolist(); selected=stable_columns(names)
    offset_col=names.index('posterior1.0_mix1.0_utility')
    assert len(names)==388 and len(selected)==313
    n=sum(int(m.sum()) for m in masks)*10
    needed=n*len(selected)*4
    if shutil.disk_usage('/tmp').free < needed+150_000_000:
        raise RuntimeError('Insufficient temporary disk for fixed tree fit')
    y=np.empty(n,np.float64); offset=np.empty(n,np.float64)
    with tempfile.TemporaryDirectory(prefix='p3_v39_') as directory:
        path=Path(directory)/'matrix.npy'
        x=np.lib.format.open_memmap(path,mode='w+',dtype=np.float32,shape=(n,len(selected)))
        pos=0
        for data,mask in zip(datasets,masks):
            events=np.flatnonzero(mask)
            for start in range(0,len(events),2000):
                rows=(events[start:start+2000,None]*10+np.arange(10)).ravel()
                raw=data['x'][rows];block=raw[:,selected].copy();block[~np.isfinite(block)]=np.nan
                x[pos:pos+len(rows)]=block;y[pos:pos+len(rows)]=data['labels'][rows]
                offset[pos:pos+len(rows)]=raw[:,offset_col];pos+=len(rows)
        x.flush();del raw,block;gc.collect()
        assert np.isfinite(offset).all() and np.all(y.reshape(-1,10).sum(1)<=1)
        first=True
        def objective(pred,data):
            nonlocal first
            if first:
                assert np.allclose(pred,offset,atol=1e-10,rtol=1e-10)
                first=False
            p=probability(pred,'event')
            return p-y,np.maximum(p*(1-p),1e-4)
        params={'objective':objective,'metric':'None','learning_rate':.025,
                'num_leaves':config['leaves'],'min_data_in_leaf':config['min_leaf'],
                'lambda_l2':config['l2'],'max_bin':127,'feature_fraction':.85,
                'bagging_fraction':.85,'bagging_freq':1,'num_threads':config['threads'],
                'seed':config['seed'],'verbosity':-1,'force_col_wise':True,'deterministic':True}
        data=lgb.Dataset(x,label=y,init_score=offset,free_raw_data=True,
                         params={k:v for k,v in params.items() if k!='objective'})
        data.construct();x._mmap.close();del x;path.unlink();gc.collect()
        start=time.time()
        def progress(env):
            if (env.iteration+1)%100==0:
                print({'round':env.iteration+1,'seconds':time.time()-start},flush=True)
        model=lgb.train(params,data,num_boost_round=config['rounds'],callbacks=[progress])
    return model,{'names':names,'selected':selected,'offset_column':offset_col,
                  'config':config,'training_events':n//10,'mode':'event'}


def predict(model,raw,spec):
    block=raw[:,spec['selected']].copy();block[~np.isfinite(block)]=np.nan
    z=model.predict(block,raw_score=True,num_threads=spec['config']['threads'])+raw[:,spec['offset_column']]
    return probability(z,'event')
