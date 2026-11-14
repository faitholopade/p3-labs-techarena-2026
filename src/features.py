import sys,gc,json,argparse,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v2/experiments'))
from common import *
from projected import state as projected_state
from tree_v1 import features as old_features
import lightgbm as lgb

def build_state(d,cut,ni,nu,root):
    fit=subset(d,np.flatnonzero(d['timestamp']<cut));folder=Path(root)/str(cut);folder.mkdir(parents=True,exist_ok=True);s=stats(fit,ni,nu)
    vectors=[projected_state(fit,ni,nu,folder,0,.5,.5,5),projected_state(fit,ni,nu,folder,0,.5,.35,0)]
    m=interaction_matrix(fit,ni,nu);counts=np.asarray(m.sum(0)).ravel();ucount=np.asarray(m.sum(1)).ravel()
    exposure=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=m.shape).tocsr();excount=np.asarray(exposure.sum(1)).ravel()
    profiles=[]
    for j,dim in [(0,8),(1,8),(1,10)]:
        raw=vectors[j][:,:dim];norm=np.linalg.norm(raw,axis=1);v=raw/np.maximum(norm[:,None],1e-8)
        users=(m@v)/np.maximum(ucount[:,None],1);neg=(exposure@v)/np.maximum(excount[:,None],1)
        profiles.append((v,norm,users,neg))
    return s,counts,ucount,profiles

def features(d,state):
    s,counts,ucount,profiles=state;ni=len(counts)-1
    base,names=old_features(d,s,sp.csr_matrix((ni+1,ni+1)),np.zeros((ni+1,0),np.float32));base=base[:,:31];names=names[:31];cols=[base]
    n=len(d['user']);c=d['items'];h=d['hist'];u=d['user'];ctx=d['context'];mask=h>0;length=mask.sum(1);age=np.maximum((d['timestamp'][:,None]-d['hist_ts'])/86400,0);hc=d['hist_ts']%86400//21600+4*((d['hist_ts']//86400+3)%7>=5)
    def add(name,x):
        x=np.asarray(x,np.float32)
        if x.ndim==1:x=np.broadcast_to(x[:,None],(n,10))
        cols.append(x.reshape(-1,1));names.append(name)
    add('item_observed_users',np.log1p(counts[c]));add('user_distinct_clicked_items',np.log1p(ucount[u]));add('history_mean_support',(np.log1p(counts[h])*mask).sum(1)/np.maximum(length,1));add('slate_support_mean',np.log1p(counts[c]).mean(1))
    for p,(v,norm,users,neg) in enumerate(profiles):
        prefix=f'pref{p}_';iv=v[c];hv=v[h];sim=np.einsum('nkd,njd->nkj',iv,hv);mean=hv.sum(1)/np.maximum(length[:,None],1);mn=np.linalg.norm(mean,axis=1);score=np.einsum('nkd,nd->nk',iv,mean)
        add(prefix+'mean',score);add(prefix+'relative',score-score.mean(1,keepdims=True));add(prefix+'to_max',score-score.max(1,keepdims=True));add(prefix+'cosine',score/np.maximum(mn[:,None],.05));add(prefix+'concentration',mn)
        add(prefix+'slate_mean',score.mean(1));add(prefix+'slate_max',score.max(1));add(prefix+'slate_std',score.std(1));add(prefix+'raw_norm',norm[c]);add(prefix+'history_norm_mean',(norm[h]*mask).sum(1)/np.maximum(length,1))
        masked=np.where(mask[:,None,:],sim,-1);sorted_=np.sort(masked,axis=2)
        add(prefix+'max',sorted_[:,:,-1]);add(prefix+'top3',sorted_[:,:,-3:].mean(2));add(prefix+'variance',(sim**2*mask[:,None,:]).sum(2)/np.maximum(length[:,None],1)-score**2)
        add(prefix+'last',sim[:,:,-1]);add(prefix+'recent3',sim[:,:,-3:].sum(2)/np.maximum(mask[:,-3:].sum(1,keepdims=True),1));add(prefix+'recent5',sim[:,:,-5:].sum(2)/np.maximum(mask[:,-5:].sum(1,keepdims=True),1))
        for label,wt in [('decay1',np.exp(-age)),('decay7',np.exp(-age/7)),('hour',(hc%4==ctx[:,None]%4)),('day',(hc//4==ctx[:,None]//4)),('reliable',counts[h]/(counts[h]+10))]:
            wt=wt*mask;value=(sim*wt[:,None,:]).sum(2)/np.maximum(wt.sum(1,keepdims=True),1e-7);add(prefix+label,value);add(prefix+label+'_relative',value-value.mean(1,keepdims=True))
        user=users[u];us=np.einsum('nkd,nd->nk',iv,user);ns=np.einsum('nkd,nd->nk',iv,neg[u]);un=np.linalg.norm(user,axis=1)
        add(prefix+'lifetime',us);add(prefix+'lifetime_relative',us-us.mean(1,keepdims=True));add(prefix+'lifetime_cosine',us/np.maximum(un[:,None],.05));add(prefix+'lifetime_norm',un)
        add(prefix+'exposure',ns);add(prefix+'exposure_relative',ns-ns.mean(1,keepdims=True));add(prefix+'profile_drift',np.linalg.norm(user-mean,axis=1))
        cc=np.einsum('nkd,njd->nkj',iv,iv);cc[:,np.arange(10),np.arange(10)]=0;add(prefix+'candidate_similarity',cc.sum(2)/9);add(prefix+'candidate_max',cc.max(2))
    return np.concatenate(cols,axis=1),names

def block(d,ix,state):
    blocks=[]
    for st in range(0,len(ix),4000):
        x,names=features(subset(d,ix[st:st+4000]),state);blocks.append(x)
    return np.concatenate(blocks),names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cache',default='v2/P3_Labs_model_and_code/cache');ap.add_argument('--cut',type=int,default=1737849600);ap.add_argument('--out',default='v3/tree');ap.add_argument('--root',default='v3/states');ap.add_argument('--rounds',type=int,default=1000);ap.add_argument('--final',action='store_true');ap.add_argument('--fixed',action='store_true');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);root=Path(a.root);root.mkdir(exist_ok=True);d,e,audit=load_data(a.cache);ni=audit['n_items'];nu=audit['n_users'];fi,vi=split_data(d,a.cut);rng=np.random.default_rng(814);srng=np.random.default_rng(808);blocks=[];ys=[];indices=[]
    for day in np.unique(d['timestamp'][fi]//86400)[2:]:
        ix=fi[d['timestamp'][fi]//86400==day]
        if len(ix)>65000:ix=np.sort(rng.choice(ix,65000,replace=False))
        ix=ix[np.sort(srng.choice(len(ix),min(45000,len(ix)),replace=False))];indices.extend(ix);ys.append(binary_labels(subset(d,ix)).ravel());file=root/f'features_{day}.npy'
        if not file.exists():
            print('Feature day',day,flush=True);state=build_state(d,int(day*86400),ni,nu,root);x,names=block(d,ix,state);np.save(file,x);(root/'names.json').write_text(json.dumps(names));del x,state;gc.collect()
        blocks.append(file)
    names=json.loads((root/'names.json').read_text());y=np.concatenate(ys);x=np.concatenate([np.load(f) for f in blocks]);pi=np.arange(len(e['user'])) if a.final else vi
    print('Prediction features',len(pi),flush=True);state=build_state(d,a.cut,ni,nu,root);xv,_=block(e if a.final else d,pi,state);del state;gc.collect();yv=None if a.final else binary_labels(subset(d,vi)).ravel();np.save(out/'xvalid.npy',xv);np.save(out/'prediction_indices.npy',pi);np.save(out/'training_indices.npy',indices);(out/'features.json').write_text(json.dumps(names))
    params=dict(objective='binary',metric='auc',learning_rate=.035,num_leaves=31,max_depth=8,min_data_in_leaf=600,lambda_l2=20,feature_fraction=.9,bagging_fraction=.85,bagging_freq=1,num_threads=4,verbosity=-1,seed=814,force_col_wise=True)
    print('Fitting',x.shape,flush=True);ds=lgb.Dataset(x,label=y,feature_name=names);dv=[] if a.final else [lgb.Dataset(xv,label=yv,reference=ds,feature_name=names)];callbacks=[lgb.log_evaluation(50)]+([] if a.final or a.fixed else [lgb.early_stopping(100)])
    model=lgb.train(params,ds,num_boost_round=a.rounds,valid_sets=dv,callbacks=callbacks);pred=model.predict(xv).reshape(-1,10);np.save(out/'pred.npy',pred);model.save_model(str(out/'model.txt'));met={} if a.final else metrics(yv,pred);met.update(rounds=model.current_iteration(),args=vars(a));(out/'metrics.json').write_text(json.dumps(met,indent=2));print(json.dumps(met),flush=True)

if __name__=='__main__':main()
