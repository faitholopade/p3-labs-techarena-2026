"""Calibration using strictly held-out base predictions and numerical history features."""
import sys,json,argparse,gc
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(BASE/'v2/experiments'),str(BASE/'v3')]
from common import *
from scipy.special import softmax
import lightgbm as lgb

def extend(x,tree,unit,pc,activity=None,repeats=None):
    z=.75*unit['user']+.25*unit['history'];p=np.clip(softmax(z,axis=1)*pc,1e-7,1-1e-7);u=np.stack([logit(p),unit['user'],unit['history'],np.repeat(z.std(1,keepdims=True),10,axis=1)],axis=2).reshape(-1,4);extra=[x,tree,u]
    if activity is not None:extra.append(np.repeat(activity,10,axis=0))
    if repeats is not None:extra.append(repeats.reshape(-1,repeats.shape[-1]))
    return np.concatenate(extra,axis=1).astype(np.float32)

def get_data():
    pack=np.load('deliverables_v2/P3_Labs_v2_model_and_code/validation_predictions.npz');blocks=[];names=json.loads(Path('v3/stack/features.json').read_text())+['tree_'+n for n in json.loads(Path('v3/tree/features.json').read_text())]+['unit_logit','unit_user','unit_history','unit_std']
    act=np.load('v4/activity/train.npy',mmap_mode='r');rep=np.load('v4/repeats/train.npy',mmap_mode='r');names+=['activity_'+n for n in json.loads(Path('v4/activity/names.json').read_text())]+['repeats_'+n for n in json.loads(Path('v4/repeats/names.json').read_text())]
    for name,tf,uf in [('earlier','tree_early','unit8_early'),('later','tree','unit8')]:
        ix=pack[name+'_indices'];x=np.load(f'v3/stack/{name}_x.npy');tree=np.load(f'v3/{tf}/xvalid.npy');unit=np.load(f'v4/{uf}/valid_logits.npz');pc=pack[name+'_v2'].sum(1,keepdims=True);blocks.append((extend(x,tree,unit,pc,act[ix],rep[ix]),pack[name+'_labels'].ravel(),ix))
    xt,yt,it=blocks[0];xv,yv,iv=blocks[1];keep=~np.isin(it,iv);return xt[np.repeat(keep,10)],yt[np.repeat(keep,10)],it[keep],xv,yv,iv,names

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='v4/flex_stack');ap.add_argument('--leaves',type=int,default=15);ap.add_argument('--rounds',type=int,default=1000);ap.add_argument('--cv',action='store_true');a=ap.parse_args();out=Path(a.out);out.mkdir(exist_ok=True,parents=True);xt,yt,it,xv,yv,iv,names=get_data();params=dict(objective='binary',metric='auc',learning_rate=.025,num_leaves=a.leaves,max_depth=4,min_data_in_leaf=2000,lambda_l2=100,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,num_threads=4,verbosity=-1,seed=731,force_col_wise=True)
    ds=lgb.Dataset(xt,label=yt,feature_name=names);dv=lgb.Dataset(xv,label=yv,reference=ds,feature_name=names);print('Calibration',xt.shape,xv.shape,flush=True);model=lgb.train(params,ds,num_boost_round=a.rounds,valid_sets=[dv],callbacks=[lgb.log_evaluation(50),lgb.early_stopping(100)]);p=model.predict(xv);model.save_model(str(out/'early_model.txt'));np.save(out/'forward_pred.npy',p);met=metrics(yv,p);met.update(rounds=model.current_iteration(),params=params);(out/'metrics.json').write_text(json.dumps(met,indent=2));(out/'features.json').write_text(json.dumps(names));print(json.dumps(met),flush=True);print(sorted(zip(names,model.feature_importance('gain')),key=lambda z:-z[1])[:25],flush=True)

if __name__=='__main__':main()
