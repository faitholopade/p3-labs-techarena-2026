import sys,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(BASE/'v2/experiments'))
from common import *
from numba import njit

NAMES=['prior_shown_log','prior_click_log','pair_rate','prior_skip_log','last_response','since_last_shown','since_pair_click','same_hour_shown_log']

@njit
def rolling(key,times,ctx,labels,order):
    out=np.zeros((len(key),8),np.float32);previous=-1;n=0;hit=0;last=0;lc=0;lasty=-1;hours=np.zeros(4)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and key[order[end]]==key[order[start]] and times[order[end]]==times[order[start]]:end+=1
        for pos in range(start,end):
            j=order[pos]
            k=key[j];t=times[j]
            if k!=previous:n=0;hit=0;last=0;lc=0;lasty=-1;hours[:]=0;previous=k
            r=out[j];r[0]=np.log1p(n);r[1]=np.log1p(hit);r[2]=(hit+.63)/(n+10);r[3]=np.log1p(n-hit);r[4]=lasty;r[5]=np.log1p((t-last)/3600) if n else -1;r[6]=np.log1p((t-lc)/3600) if lc else -1;r[7]=np.log1p(hours[ctx[j]%4])
        for pos in range(start,end):
            j=order[pos];t=times[j]
            if labels[j]>=0:
                n+=1;hit+=labels[j];last=t;lasty=labels[j];hours[ctx[j]%4]+=1
                if labels[j]:lc=t
        start=end
    return out

def main():
    d,e,a=load_data('v2/P3_Labs_model_and_code/cache');items=np.r_[d['items'],e['items']];users=np.r_[d['user'],e['user']];key=(users.astype(np.int64)[:,None]*(a['n_items']+1)+items).ravel();t=np.repeat(np.r_[d['timestamp'],e['timestamp']],10);c=np.repeat(np.r_[d['context'],e['context']],10);y=np.r_[binary_labels(d).ravel(),np.full(len(e['user'])*10,-1)];order=np.lexsort((t,key));x=rolling(key,t,c,y,order).reshape(-1,10,8);out=Path('v4/repeats');out.mkdir(exist_ok=True);np.save(out/'train.npy',x[:len(d['user'])]);np.save(out/'evaluation.npy',x[len(d['user']):]);(out/'names.json').write_text(json.dumps(NAMES));fi,vi=split_data(d,1737849600);print('Prior exposures',float((x[vi,:,0]>0).mean()),'eval',float((x[len(d['user']):,:,0]>0).mean()),flush=True)

if __name__=='__main__':main()
