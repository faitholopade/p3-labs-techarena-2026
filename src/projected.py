"""Learn preference factors after removing exposure-driven item directions."""
import sys,json,time,argparse
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'v2/experiments'))
from common import *
from topic import counts
from scipy.linalg import qr
from scipy.special import softmax

def state(fit,ni,nu,folder,power=0,uexp=.5,iexp=.35,reg=0,augment=None):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    name=f'p{power}_u{uexp}_i{iexp}_r{reg}'+('_augment' if augment is not None else '')
    if (folder/(name+'.npy')).exists():return np.load(folder/(name+'.npy'))
    imp=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=(nu+1,ni+1)).tocsr();imp.data=np.sqrt(imp.data);ip0=np.asarray(imp.sum(0)).ravel();up0=np.asarray(imp.sum(1)).ravel()
    if (folder/'exposure.npz').exists():q0=np.load(folder/'exposure.npz')['v']
    else:
        _,ss,vt=randomized_svd(sp.diags(1/np.maximum(up0,1)**.5)@imp@sp.diags(1/np.maximum(ip0,1)**.5),n_components=33,n_iter=15,random_state=2718);q0=vt.T;np.savez_compressed(folder/'exposure.npz',v=q0,s=ss)
    if augment is not None:
        extra=dict(augment);extra['target']=np.full(len(extra['user']),10,np.int64);fit={k:np.concatenate([fit[k],extra[k]]) for k in fit}
    m=interaction_matrix(fit,ni,nu) if power==0 else counts(fit,ni,nu)
    m.data=m.data**power if power else np.ones_like(m.data)
    ip=np.asarray(m.sum(0)).ravel();up=np.asarray(m.sum(1)).ravel()
    w=sp.diags(1/np.maximum(up+reg,1)**uexp)@m@sp.diags(1/np.maximum(ip+reg,1)**iexp)
    q=qr(q0[:,:17]*np.sqrt(ip0[:,None])/np.maximum(ip[:,None]+reg,1)**iexp,mode='economic')[0].astype(np.float32)
    rng=np.random.default_rng(2718);v=rng.normal(size=(ni+1,32)).astype(np.float32);v-=q@(q.T@v)
    for _ in range(25):
        u=qr(w@v,mode='economic')[0];v=w.T@u;v-=q@(q.T@v);v=qr(v,mode='economic')[0]
    z=u.T@w;z-=(z@q)@q.T;_,ss,vt=np.linalg.svd(z,full_matrices=False);v=vt.T.astype(np.float32);v[0]=0;np.save(folder/(name+'.npy'),v);np.savez_compressed(folder/(name+'_stats.npz'),s=ss,item_counts=ip,user_counts=up)
    return v

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default='v3/projected_sweep');ap.add_argument('--cut',type=int,default=1737849600);ap.add_argument('--augment',action='store_true');a=ap.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);cache='v2/P3_Labs_model_and_code/cache';d,e,audit=load_data(cache);fi,vi=split_data(d,a.cut);fit=subset(d,fi);val=subset(d,vi);ni=audit['n_items'];nu=audit['n_users'];y=binary_labels(val)
    pc=np.load('v2/experiments/n32/valid_pred.npy').sum(1,keepdims=True) if a.cut==1737849600 else np.load('v2/experiments/check_jan25/event_pred.npy').reshape(-1,1)
    baseline=np.load('deliverables_v2/P3_Labs_v2_model_and_code/validation_predictions.npz')['later_v2' if a.cut==1737849600 else 'earlier_v2'];result=[]
    variants=[(0,.5,.35,0),(0,.25,.35,0),(0,.75,.35,0),(0,.5,.2,0),(0,.5,.5,0),(0,.5,.5,5),(.5,.5,.35,0),(1,.5,.35,0),(1,.5,.5,5)]
    for power,ue,ie,reg in variants:
        v=state(fit,ni,nu,out,power,ue,ie,reg,val if a.augment else None)
        for dim in [8,10]:
            vv=v[:,:dim];vv=vv/np.maximum(np.linalg.norm(vv,axis=1,keepdims=True),1e-6);hm=vv[val['hist']].sum(1)/np.maximum((val['hist']>0).sum(1,keepdims=True),1);sim=np.einsum('nkd,nd->nk',vv[val['items']],hm);best=(0,0,None)
            for temp in [3,4,5,6]:
                p=softmax(sim*temp,axis=1)*pc;auc=roc_auc_score(y.ravel(),p.ravel())
                if auc>best[0]:best=(auc,temp,p)
            name=f'p{power}_u{ue}_i{ie}_r{reg}_d{dim}';np.save(out/(name+'_pred.npy'),best[2]);mix=roc_auc_score(y.ravel(),(.5*best[2]+.5*baseline).ravel());rec={'power':power,'ue':ue,'ie':ie,'reg':reg,'dim':dim,'auc':best[0],'temperature':best[1],'blend_auc':mix};result.append(rec);print(rec,flush=True)
    (out/'results.json').write_text(json.dumps(result,indent=2))

if __name__=='__main__':main()
