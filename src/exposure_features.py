"""Additional candidate/history geometry in the learned exposure subspace."""
import sys,json,argparse
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'v2/experiments'))
from common import *

def state(fit,ni,nu,cut,basis=None):
    q=np.load(basis if basis else BASE/'v3/states'/str(cut)/'exposure.npz')['v'];m=interaction_matrix(fit,ni,nu);n=np.asarray(m.sum(1)).ravel();ex=sp.coo_matrix((np.ones(len(fit['user'])*10,np.float32),(np.repeat(fit['user'],10),fit['items'].ravel())),shape=m.shape).tocsr();en=np.asarray(ex.sum(1)).ravel();result=[]
    for dim in [8,16,32]:
        raw=q[:,1:dim+1].copy();norm=np.linalg.norm(raw,axis=1);v=raw/np.maximum(norm[:,None],1e-8);v[0]=0;uc=m@v/np.maximum(n[:,None],1);ue=ex@v/np.maximum(en[:,None],1);result.append((v,norm,uc,ue))
    return result

def features(d,s):
    n=len(d['user']);c=d['items'];h=d['hist'];u=d['user'];mask=h>0;length=mask.sum(1);cols=[];names=[]
    def add(name,x):
        x=np.asarray(x,np.float32)
        if x.ndim==1:x=np.repeat(x[:,None],10,axis=1)
        cols.append(x.reshape(-1));names.append(name)
    for k,(v,norm,uc,ue) in enumerate(s):
        pref='exp'+str(k)+'_';iv=v[c];hv=v[h];hist=hv.sum(1)/np.maximum(length[:,None],1);ch=uc[u];ex=ue[u]
        for label,profile in [('history',hist),('clicked',ch),('shown',ex),('history_residual',hist-ex),('clicked_residual',ch-ex)]:
            normp=np.linalg.norm(profile,axis=1);score=np.einsum('nkd,nd->nk',iv,profile);add(pref+label,score);add(pref+label+'_relative',score-score.mean(1,keepdims=True));add(pref+label+'_cos',score/np.maximum(normp[:,None],.05));add(pref+label+'_norm',normp)
        add(pref+'item_norm',norm[c]);sim=np.einsum('nkd,njd->nkj',iv,hv);add(pref+'history_max',np.where(mask[:,None,:],sim,-1).max(2));cs=np.einsum('nkd,njd->nkj',iv,iv);add(pref+'candidate_similarity',(cs.sum(2)-1)/9)
    return np.stack(cols,1),names

def main():
    out=Path('v5/exposure');out.mkdir(exist_ok=True);d,e,a=load_data('v2/P3_Labs_model_and_code/cache')
    for name,cut in [('earlier',1737763200),('later',1737849600)]:
        fi,vi=split_data(d,cut);s=state(subset(d,fi),a['n_items'],a['n_users'],cut);parts=[]
        for start in range(0,len(vi),4096):xx,names=features(subset(d,vi[start:start+4096]),s);parts.append(xx)
        x=np.concatenate(parts);np.save(out/(name+'.npy'),x);print(name,x.shape,flush=True)
    (out/'features.json').write_text(json.dumps(names))

if __name__=='__main__':main()
