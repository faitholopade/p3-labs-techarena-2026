"""Broaden warm-up negative sampling with pools of similar exposure profiles."""
import sys,os
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'v2/experiments'))
from common import *
from sklearn.cluster import MiniBatchKMeans

def pooled(fit,ni,nu,cut,clusters=64,basis=None,cache=None):
    folder=Path(cache) if cache else BASE/'v5';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'exposure_pools_{cut}_{clusters}.npz'
    if path.exists():return dict(np.load(path))
    imp=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=(nu+1,ni+1)).tocsr();imp.data=np.sqrt(imp.data);ip=np.asarray(imp.sum(0)).ravel();z=np.load(basis if basis else BASE/'v3/states'/str(cut)/'exposure.npz');v=z['v'][:,1:17]/np.sqrt(np.maximum(ip[:,None],1));u=imp@v;u/=np.maximum(np.linalg.norm(u,axis=1,keepdims=True),1e-8);valid=np.flatnonzero(np.diff(imp.indptr)>0);km=MiniBatchKMeans(clusters,random_state=2718,batch_size=4096,n_init=3,max_iter=150);km.fit(u[valid]);labels=km.predict(u);groups=np.repeat(labels[fit['user']],10);order=np.argsort(groups,kind='stable');pool=fit['items'].ravel()[order];ptr=np.r_[0,np.cumsum(np.bincount(groups,minlength=clusters))];result={'labels':labels,'pool':pool,'ptr':ptr};temporary=path.with_suffix('.'+str(os.getpid())+'.npz');np.savez_compressed(temporary,**result);os.replace(temporary,path);print('Built exposure pools',clusters,flush=True);return result

def sample(pool,users,rng,n=9):
    groups=pool['labels'][users];ptr=pool['ptr'];loc=ptr[groups,None]+(rng.random((len(users),n))*(ptr[groups+1]-ptr[groups])[:,None]).astype(np.int64);return pool['pool'][loc]
