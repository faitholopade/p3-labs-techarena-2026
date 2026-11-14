import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v2/experiments'))
from common import *
from scipy.special import softmax
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def meta_features(old,tree,user,hist,xnew,names):
    n=len(old);cols=[];labels=[]
    def add(label,x):
        x=np.asarray(x,np.float32)
        if x.ndim==1:x=np.broadcast_to(x[:,None],(n,10))
        cols.append(x.reshape(-1));labels.append(label)
    pc=old.sum(1,keepdims=True)
    neural=softmax(.75*user+.25*hist,axis=1)*pc
    for label,p in [('old',old),('tree',tree),('neural',neural),('user',softmax(user,axis=1)*pc),('history',softmax(hist,axis=1)*pc)]:add('logit_'+label,logit(np.clip(p,1e-7,1-1e-7)))
    add('user_score',user);add('history_score',hist);add('user_score_mean',user.mean(1));add('user_score_max',user.max(1));add('user_score_std',user.std(1));add('history_score_mean',hist.mean(1));add('history_score_max',hist.max(1));add('history_score_std',hist.std(1));add('old_event',pc[:,0]);add('tree_event',tree.sum(1))
    chosen=['item_logit','user_event_logit','history_length_scaled','last_click_hours_log','user_events_log','item_clicks_log','item_observed_users','user_distinct_clicked_items','pref0_concentration','pref0_raw_norm','pref0_variance','pref0_max','pref0_candidate_similarity','pref0_exposure','pref0_lifetime_norm','pref0_profile_drift']
    for name in chosen:
        if name in names:add(name,xnew[:,names.index(name)].reshape(-1,10))
    return np.stack(cols,1),labels

def main():
    root=Path('v3/stack');root.mkdir(exist_ok=True);pack=np.load('deliverables_v2/P3_Labs_v2_model_and_code/validation_predictions.npz');blocks=[]
    for name,treefolder,nnfolder in [('earlier','tree_early','latent8_early'),('later','tree','latent8_joint')]:
        old=pack[name+'_v2'];tree=np.load(f'v3/{treefolder}/pred.npy');z=np.load(f'v3/{nnfolder}/valid_logits.npz');xx=np.load(f'v3/{treefolder}/xvalid.npy');names=json.loads(Path(f'v3/{treefolder}/features.json').read_text());x,labels=meta_features(old,tree,z['user'],z['history'],xx,names);y=pack[name+'_labels'].ravel();ix=pack[name+'_indices'];blocks.append((x,y,ix));np.save(root/(name+'_x.npy'),x)
    xt,yt,it=blocks[0];xv,yv,iv=blocks[1];keep=~np.isin(it,iv);keep=np.repeat(keep,10);xt=xt[keep];yt=yt[keep]
    print('Meta fit rows',len(yt),'validation rows',len(yv),flush=True);base=.45*np.load('v3/tree/pred.npy')+.25*expit(xv[:,2]).reshape(-1,10)+.3*pack['later_v2'];print('Fixed blend later',roc_auc_score(yv,base.ravel()),flush=True)
    sc=StandardScaler();xx=sc.fit_transform(xt[:,:5]);reg=LogisticRegression(C=.1,max_iter=300).fit(xx,yt);p=reg.predict_proba(sc.transform(xv[:,:5]))[:,1];print('Logistic',metrics(yv,p),'coeff',reg.coef_,flush=True);np.save(root/'linear_pred.npy',p)
    params=dict(objective='binary',metric='auc',learning_rate=.025,num_leaves=7,max_depth=3,min_data_in_leaf=2000,lambda_l2=100,feature_fraction=1,bagging_fraction=.9,bagging_freq=1,num_threads=4,verbosity=-1,seed=731,force_col_wise=True)
    train=lgb.Dataset(xt,label=yt,feature_name=labels);valid=lgb.Dataset(xv,label=yv,reference=train,feature_name=labels);model=lgb.train(params,train,num_boost_round=500,valid_sets=[valid],callbacks=[lgb.log_evaluation(50),lgb.early_stopping(60)]);p=model.predict(xv);met=metrics(yv,p);met['rounds']=model.current_iteration();met['params']=params;model.save_model(str(root/'early_model.txt'));np.save(root/'pred.npy',p);(root/'features.json').write_text(json.dumps(labels));(root/'metrics.json').write_text(json.dumps(met,indent=2));print('Meta',met,flush=True)
    # Preserve the precise earlier-only training mask and all source event identities.
    np.savez_compressed(root/'validation.npz',earlier_keep=keep,earlier_indices=it,later_indices=iv,later_labels=yv,later_prediction=p)

if __name__=='__main__':main()
