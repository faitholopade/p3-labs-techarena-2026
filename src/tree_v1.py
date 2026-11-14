"""An independent gradient-boosting model with forward-only aggregate features."""
import argparse,json,time,gc
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.special import logit
import lightgbm as lgb
from common import *

def state_for(d,cut,ni,nu,cache,dim=64):
    out=Path(cache)/('derived_'+str(cut));out.mkdir(exist_ok=True)
    fit=subset(d,np.flatnonzero(d['timestamp']<cut));s=stats(fit,ni,nu)
    vname='svd.npy' if dim==64 else 'svd'+str(dim)+'.npy'
    if not (out/'co.npz').exists() or not (out/vname).exists():
        m=interaction_matrix(fit,ni,nu)
        if not (out/'co.npz').exists():sp.save_npz(out/'co.npz',coclick(m))
        if not (out/vname).exists():np.save(out/vname,init_svd(m,dim))
    return s,sp.load_npz(out/'co.npz').tocsr(),np.load(out/vname)

def features(d,s,co,v):
    row,event,age=feature_arrays(d,s,False);n=len(d['user']);c=d['items'];h=d['hist'];mask=h>0;length=mask.sum(1);ctx=d['context'];ts=d['timestamp'];cols=[];names=[]
    def add(name,x):
        x=np.asarray(x,dtype=np.float32)
        if x.ndim==1:x=np.broadcast_to(x[:,None],(n,10))
        cols.append(x.reshape(-1));names.append(name)
    for j,name in enumerate(['item_logit','item_context_delta','repeat_count_log','is_last_click','repeat_age_log','item_impressions_log']):add(name,row[:,:,j])
    for j,name in enumerate(['user_event_logit','history_length_scaled','last_click_hours_log','user_events_log']):add(name,event[:,j])
    add('context',ctx);add('hour',(ts%86400)/3600)
    add('history_unique',np.array([len(set(r[r>0])) for r in h],dtype=np.float32))
    span=np.maximum(d['hist_ts'].max(1)-np.where(mask,d['hist_ts'],ts[:,None]).min(1),0)
    add('history_span_log',np.where(length>0,np.log1p(span/3600),np.nan))
    for days in [1/24,0.25,1,3,7]:add('history_clicks_'+str(days),((age<days)&mask).sum(1))
    for k in [5,20,100,500]:
        rate=(s['hit'][c]+k*s['g'])/(s['imp'][c]+k)
        add('item_logit_k'+str(k),logit(np.clip(rate,1e-6,1-1e-6)))
    add('item_clicks_log',np.log1p(s['hit'][c]));add('item_context_impressions_log',np.log1p(s['ci'][c,ctx[:,None]]))
    il=row[:,:,0];add('item_logit_minus_slate_mean',il-il.mean(1,keepdims=True));add('item_logit_minus_slate_max',il-il.max(1,keepdims=True));add('item_logit_rank',(il[:,:,None]>il[:,None,:]).sum(2));add('slate_item_std',il.std(1))
    hp=(s['hit'][h]+50*s['g'])/(s['imp'][h]+50)
    add('history_mean_ctr',(hp*mask).sum(1)/np.maximum(length,1));add('history_last_ctr',hp[:,-1])
    for dim in [x for x in [8,16,32,64,128,256] if x<=v.shape[1]]:
        vv=v[:,:dim];vv=vv/np.maximum(np.linalg.norm(vv,axis=1,keepdims=True),1e-6)
        hv=vv[h];iv=vv[c]
        mean=(hv*mask[:,:,None]).sum(1)/np.maximum(length[:,None],1)
        sim=np.einsum('nkd,nd->nk',iv,mean)
        add('svd_mean_'+str(dim),sim);add('svd_relative_'+str(dim),sim-sim.mean(1,keepdims=True))
        if dim==32:
            add('svd_last',np.einsum('nkd,nd->nk',iv,hv[:,-1]))
            recent=(hv[:,-3:]*mask[:,-3:,None]).sum(1)/np.maximum(mask[:,-3:].sum(1)[:,None],1)
            add('svd_recent3',np.einsum('nkd,nd->nk',iv,recent))
        if dim>=64:
            simhist=np.einsum('nkd,njd->nkj',iv,hv)
            simhist=np.where(mask[:,None,:],simhist,-1)
            top=np.sort(simhist,axis=2)[:,:,-3:]
            add('svd_max_'+str(dim),top[:,:,-1]);add('svd_top3_'+str(dim),np.maximum(top,0).mean(2))
            add('svd_max_relative_'+str(dim),top[:,:,-1]-top[:,:,-1].mean(1,keepdims=True))
            weighted=np.exp(-age)*mask;recent=(hv*weighted[:,:,None]).sum(1)/np.maximum(weighted.sum(1)[:,None],1e-6)
            add('svd_decay_'+str(dim),np.einsum('nkd,nd->nk',iv,recent))
    cf_parts=[]
    for st in range(0,n,512):
        cc=c[st:st+512];hh=h[st:st+512]
        aa=np.broadcast_to(cc[:,:,None],(len(cc),10,20)).reshape(-1);bb=np.broadcast_to(hh[:,None,:],(len(cc),10,20)).reshape(-1)
        cf_parts.append(np.asarray(co[aa,bb]).reshape(len(cc),10,20))
    cf=np.concatenate(cf_parts)
    add('cf_sum',cf.sum(2));add('cf_mean',cf.sum(2)/np.maximum(length[:,None],1));add('cf_max',cf.max(2));add('cf_last',cf[:,:,-1]);add('cf_recent3',cf[:,:,-3:].sum(2));add('cf_recency',(cf*np.exp(-age[:,None,:])).sum(2))
    add('cf_relative',cf.sum(2)-cf.sum(2).mean(1,keepdims=True))
    return np.stack(cols,axis=1),names

def build_block(d,ix,s,co,v):
    chunks=[]
    for st in range(0,len(ix),10000):
        xx,names=features(subset(d,ix[st:st+10000]),s,co,v);chunks.append(xx)
    return np.concatenate(chunks),names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='p3labs/cache');ap.add_argument('--out',default='p3labs/tree_a');ap.add_argument('--cutoff',type=int,default=1737763200);ap.add_argument('--threads',type=int,default=4);ap.add_argument('--rounds',type=int,default=700);ap.add_argument('--dim',type=int,default=256);ap.add_argument('--fixed',action='store_true');ap.add_argument('--final',action='store_true');a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True);d,e,audit=load_data(a.cache);ni=audit['n_items'];nu=audit['n_users'];fitix,valix=split_data(d,a.cutoff);rng=np.random.default_rng(814)
    if a.final:
        if a.cutoff<=d['timestamp'].max():raise ValueError('Final cutoff must follow every training event')
        valix=np.arange(len(e['user']))
    prediction_data=e if a.final else d
    days=np.unique(d['timestamp'][fitix]//86400);blocks=[];targets=[];source=[]
    for day in days[2:]:
        cut=int(day*86400);ix=fitix[d['timestamp'][fitix]//86400==day]
        if len(ix)>65000:ix=np.sort(rng.choice(ix,65000,replace=False))
        print('Building forward-only training block',day,len(ix),flush=True)
        s,co,v=state_for(d,cut,ni,nu,a.cache,a.dim);x,names=build_block(d,ix,s,co,v)
        blocks.append(x);targets.append(binary_labels(subset(d,ix)).ravel());source.extend(ix);del co,v;gc.collect()
    x=np.concatenate(blocks);y=np.concatenate(targets);del blocks;gc.collect()
    print('Prediction features',len(valix),flush=True);s,co,v=state_for(d,a.cutoff,ni,nu,a.cache,a.dim);xv,names=build_block(prediction_data,valix,s,co,v);yv=None if a.final else binary_labels(subset(d,valix)).ravel();del co,v;gc.collect()
    if not a.final:
        np.save(out/'validation_indices.npy',valix);np.save(out/'xvalid.npy',xv);np.save(out/'yvalid.npy',yv)
    np.savez_compressed(out/'stats.npz',**s)
    (out/'features.json').write_text(json.dumps(names))
    params=dict(objective='binary',metric='auc',learning_rate=.04,num_leaves=31,max_depth=8,min_data_in_leaf=600,lambda_l2=20,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,num_threads=a.threads,verbosity=-1,seed=814,force_col_wise=True)
    print('Training tree',x.shape,'validation',xv.shape,flush=True)
    train=lgb.Dataset(x,label=y,feature_name=names)
    validsets=[] if a.final else [lgb.Dataset(xv,label=yv,reference=train,feature_name=names)]
    callbacks=[lgb.log_evaluation(50)]+([] if a.final or a.fixed else [lgb.early_stopping(80)])
    model=lgb.train(params,train,num_boost_round=a.rounds,valid_sets=validsets,callbacks=callbacks)
    p=model.predict(xv).reshape(-1,10);met={} if a.final else metrics(yv,p);met.update(best_iteration=model.current_iteration(),training_rows=len(y),prediction_events=len(valix),cutoff=a.cutoff,config=vars(a))
    met['svd_dim']=a.dim
    print(json.dumps(met),flush=True);np.save(out/('evaluation_pred.npy' if a.final else 'valid_pred.npy'),p);model.save_model(str(out/'model.txt'));(out/'metrics.json').write_text(json.dumps(met,indent=2));(out/'params.json').write_text(json.dumps(params,indent=2))
    np.save(out/'training_indices.npy',np.array(source))
    if not a.final:np.save(out/'xtrain.npy',x);np.save(out/'ytrain.npy',y)

if __name__=='__main__':main()
