"""Nonlinear latent-coordinate residuals with causal daily factor inputs."""
import argparse,gc,hashlib,importlib.util,json,math,time,sys
from pathlib import Path
import numpy as np,torch,lightgbm as lgb
from scipy.linalg import orthogonal_procrustes
from scipy.special import softmax,logit,expit
from sklearn.metrics import roc_auc_score,log_loss
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runtime/v145';sys.path.insert(0,str(ROOT/'src'));from uncertainty import numpy_direction
CUTS=[1737590400,1737676800,1737763200,1737849600,1737936000]
FACTORS=['runtime/benchmark/inputs/factors/thursday.pt','runtime/benchmark/inputs/factors/friday.pt','runtime/v17/validation/factors/earlier_factor_users_only.pt','runtime/v17/validation/factors/later_factor_users_only.pt','runtime/v17/models/factor_users_only.pt']
def write(p,obj):Path(p).write_text(json.dumps(obj,indent=2)+'\n')
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def first(d,lo,hi):
    ix=np.flatnonzero((d['timestamp']>=lo)&(d['timestamp']<hi));ix=ix[np.argsort(d['timestamp'][ix],kind='stable')];_,j=np.unique(d['user'][ix],return_index=True);return np.sort(ix[j])
def arrays(s):
    s={k:v.numpy() for k,v in s.items()};im=s['item.weight'];um=s['user.weight'];is_=np.exp(s['item_logstd.weight'].clip(-3,1));us=np.exp(s['user_logstd.weight'].clip(-3,1));iv=numpy_direction(im,is_);uv=numpy_direction(um,us);return dict(iv=iv,uv=uv,q=s['exposure_items'],v=s['exposure_user.weight'],ib=s['item_bias.weight'][:,0],outside=s['outside.weight'][:,0],inorm=np.linalg.norm(im,axis=1),unorm=np.linalg.norm(um,axis=1),istd=is_[:,0],ustd=us[:,0],estd=np.exp(s['exposure_logstd.weight'].clip(-4,1))[:,0],beta=float(np.exp(np.clip(s['log_beta'],math.log(2),math.log(8)))),eb=float(np.exp(np.clip(s['log_exposure_beta'],math.log(.5),math.log(8))))/math.sqrt(8))

def features(day,forecast=False):
    name='forecast' if forecast else f'day{day}';path=OUT/(name+'.npz')
    if path.exists():return path
    started=time.time();d=np.load(ROOT/('runtime/v11/data/evaluation.npz' if day==4 else 'runtime/v11/data/train.npz'));training=np.load(ROOT/'runtime/v11/data/train.npz');cut=CUTS[day];past=training['timestamp']<cut;c0=training['items'][past];u0=training['user'][past];y0=training['target'][past];s=arrays(torch.load(ROOT/FACTORS[day],weights_only=False,map_location='cpu')['state']);anchor=arrays(torch.load(ROOT/FACTORS[0],weights_only=False,map_location='cpu')['state']);rp,_=orthogonal_procrustes(s['iv'][1:],anchor['iv'][1:]);re,_=orthogonal_procrustes(s['q'][1:],anchor['q'][1:]);iv=s['iv']@rp;uv=s['uv']@rp;q=s['q']@re;eu=s['v']@re;ni=len(iv);nu=len(uv);uc=np.bincount(u0,minlength=nu);uh=np.bincount(u0,weights=y0<10,minlength=nu);ic=np.bincount(c0.ravel(),minlength=ni);hit=np.zeros_like(c0,dtype=float);clicked=y0<10;hit[np.flatnonzero(clicked),y0[clicked]]=1;ih=np.bincount(c0.ravel(),weights=hit.ravel(),minlength=ni)
    if day==4:idx=np.arange(len(d['user']))
    elif forecast:idx=np.load(ROOT/'runtime/v142/primary_evidence.npz')['indices']
    else:idx=first(d,cut,cut+86400)
    items=d['items'][idx];users=d['user'][idx];hist=d['hist'][idx];ht=d['hist_ts'][idx];ts=d['timestamp'][idx];ctx=d['context'][idx];assert np.all(ht[hist>0]<cut),name;parts=[];initial=[];width=None
    for st in range(0,len(idx),1024):
        sl=slice(st,st+1024);c=items[sl];u=users[sl];h=hist[sl];mask=h>0;N=len(u);ii=iv[c];uu=uv[u];qq=q[c];ee=eu[u];pref=np.einsum('nkd,nd->nk',ii,uu)*s['beta'];exposure=np.einsum('nkd,nd->nk',qq,ee)*s['eb'];z=pref+exposure+s['ib'][c]+s['outside'][u,None];p=softmax(np.c_[z,np.zeros(N)],axis=1)[:,:10];init=logit(p.clip(1e-6,1-1e-6));initial.append(init.ravel());scalar=[init,pref,exposure,s['ib'][c],s['inorm'][c],s['istd'][c],np.log1p(ic[c]),(ih[c]+5)/(ic[c]+80)]
        events=[s['outside'][u],s['unorm'][u],s['ustd'][u],s['estd'][u],np.linalg.norm(ee,axis=1),np.log1p(uc[u]),(uh[u]+3)/(uc[u]+5),mask.sum(1),((ts[sl]%86400)/86400),ctx[sl]%4,ctx[sl]//4]
        for score in [z,pref,exposure]:
            scalar.extend([score-score.mean(1,keepdims=True),score-score.max(1,keepdims=True)]);events.extend([score.mean(1),score.max(1),score.std(1)])
        hp=(iv[h]*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1);he=(q[h]*mask[:,:,None]).sum(1)/np.maximum(mask.sum(1,keepdims=True),1);scalar.extend([np.einsum('nkd,nd->nk',ii,hp),np.einsum('nkd,nd->nk',qq,he)]);cos=np.einsum('nkd,njd->nkj',ii,iv[h]);scalar.append(np.where(mask[:,None],cos,-1).max(2));scalar.extend([(c[:,:,None]==h[:,None,:]).sum(2),(c==h[:,-1,None]).astype(float)]);events.extend([np.log1p(np.maximum(ts[sl]-ht[sl,-1],0)/3600)*(mask.sum(1)>0),np.linalg.norm(hp,axis=1),np.linalg.norm(he,axis=1)])
        base=np.stack(scalar+[np.broadcast_to(x[:,None],(N,10)) for x in events],axis=2).astype(np.float32);width=base.shape[-1];raw=np.concatenate([ii,np.broadcast_to(uu[:,None],ii.shape),qq,np.broadcast_to(ee[:,None],qq.shape)],axis=2);outer=np.concatenate([(ii[:,:,:,None]*uu[:,None,None,:]).reshape(N,10,64),(qq[:,:,:,None]*ee[:,None,None,:]).reshape(N,10,64)],axis=2);parts.append(np.concatenate([base,raw,outer],axis=2).reshape(-1,width+160).astype(np.float32))
    x=np.concatenate(parts);labels=((np.arange(10)[None]==d['target'][idx,None]).astype(np.float32).ravel() if day<4 else np.array([],np.float32));tmp=path.with_suffix('.writing.npz');np.savez_compressed(tmp,X=x,init=np.concatenate(initial),indices=idx,labels=labels,control_width=width);tmp.replace(path);write(OUT/(name+'_features.json'),dict(day=day,forecast=forecast,cutoff=cut,queries=len(idx),features=x.shape[1],scalar_features=width,history_strictly_before_cutoff=True,maximum_training_timestamp=int(training['timestamp'][past].max()),minimum_query_timestamp=int(ts.min()),factor_sha256=sha(ROOT/FACTORS[day]),source_sha256=sha(Path(__file__)),seconds=time.time()-started));print(dict(stage='features',name=name,shape=x.shape,seconds=time.time()-started),flush=True);return path

def fit(cohort):
    days=list(range(cohort+1)) if cohort<3 else list(range(4));folder=OUT/f'cohort{cohort}';folder.mkdir(parents=True,exist_ok=True);data=[np.load(features(d)) for d in days];x=np.concatenate([d['X'] for d in data]);y=np.concatenate([d['labels'] for d in data]);init=np.concatenate([d['init'] for d in data]);width=int(data[0]['control_width']);records=[]
    for leaves in [7,15]:
        for arm,xx in [('control',x[:,:width]),('latent',x)]:
            start=time.time();dataset=lgb.Dataset(xx,label=y,init_score=init,free_raw_data=True);params=dict(objective='binary',metric='binary_logloss',num_leaves=leaves,max_depth=4,min_data_in_leaf=2000,lambda_l2=20,learning_rate=.03,num_threads=3,verbosity=-1,deterministic=True,force_col_wise=True,seed=145026,feature_pre_filter=False);model=lgb.train(params,dataset,num_boost_round=250);model.save_model(str(folder/f'{arm}{leaves}.txt'));r=dict(arm=arm,leaves=leaves,training_days=days,rows=len(y),features=xx.shape[1],seconds=time.time()-start);records.append(r);print(r,flush=True);del dataset,model;gc.collect()
    write(folder/'fit.json',dict(records=records,training_days=days,source_sha256=sha(Path(__file__)),plan_sha256=sha(ROOT/'experiments/v145/plan.json'),lightgbm_version=lgb.__version__,query_outcomes_used=False))

def predict(cohort):
    f=np.load(features(1,True) if cohort==0 else features(4 if cohort==3 else cohort+1));x=f['X'];width=int(f['control_width']);values={}
    for leaves in [7,15]:
        predictions={arm:lgb.Booster(model_file=str(OUT/f'cohort{cohort}/{arm}{leaves}.txt')).predict(x[:,:width] if arm=='control' else x,raw_score=True,num_threads=3) for arm in ['control','latent']};values[str(leaves)]=predictions['latent']-predictions['control']
    np.savez_compressed(OUT/f'cohort{cohort}/predictions.npz',indices=f['indices'],**values)

def transfer(base,delta,mode,strength):
    d=delta.reshape(-1,10);b=base.reshape(-1,10)
    if mode=='choice':d=d-d.mean(1,keepdims=True)
    d=.25*np.tanh(d/.25)
    if mode=='choice':return (softmax(np.log(b)+strength*d,axis=1)*b.sum(1,keepdims=True)).ravel()
    return expit(logit(b)+strength*d).ravel()

def screen():
    e=np.load(ROOT/'runtime/v142/primary_evidence.npz');v=np.load(OUT/'cohort0/predictions.npz');assert np.array_equal(e['indices'],v['indices']);base=e['candidate'];y=e['labels'];ts=np.load(ROOT/'runtime/v11/data/train.npz')['timestamp'][e['indices']];masks=dict(all=np.ones(len(y),bool),first_day=np.repeat(ts<1737763200,10),delayed=np.repeat(ts>=1737763200,10),third_day=np.repeat(ts>=1737849600,10));ba={k:roc_auc_score(y[z],base[z]) for k,z in masks.items()};rows=[];pred={}
    for leaves in [7,15]:
        for mode in ['choice','binary']:
            for strength in [.25,.5,1.]:
                p=transfer(base,v[str(leaves)],mode,strength);g={k:float(roc_auc_score(y[z],p[z])-ba[k]) for k,z in masks.items()};ll=float(log_loss(y,p)-log_loss(y,base));row=dict(leaves=leaves,mode=mode,strength=strength,gains=g,logloss_change=ll,eligible=g['all']>=.00005 and min(g.values())>=0 and ll<=0);rows.append(row);pred[f'{leaves}_{mode}_{strength}']=p;print(row,flush=True)
    write(OUT/'primary_screen.json',dict(results=rows,baseline_auc=ba));np.savez_compressed(OUT/'primary_predictions.npz',indices=e['indices'],**pred)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['run','screen'],required=True);p.add_argument('--cohort',type=int);a=p.parse_args();torch.set_num_threads(3)
    if a.stage=='run':fit(a.cohort);predict(a.cohort)
    else:screen()
