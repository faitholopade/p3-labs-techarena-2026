import sys,json
from pathlib import Path
BASE=Path(__file__).resolve().parents[1];sys.path[:0]=[str(BASE/'v2/experiments'),str(BASE/'v3')]
from common import *
from numba import njit

NAMES=['user_count','user_rate3','user_rate10','user_rate30','user_beta_variance','user_hour_rate2','user_hour_rate10','user_day_rate2','user_day_rate10','user_context_rate2','user_context_rate10','prev_clicked','prev2_clicked','last3_rate','last5_rate','last10_rate','last20_rate','ewma2','ewma5','log_since_first','log_since_event','log_since_clicked_event','log_since_skip','last_run','mean_log_gap','std_log_gap','activity_rate','relative_gap','log_observation_days','last_hour_count','last_day_count','last_context_count']

@njit
def rolling(user,timestamp,context,target,order,nu):
    result=np.zeros((len(user),32),np.float32);count=np.zeros(nu+1,np.int64);hit=np.zeros(nu+1);cn=np.zeros((nu+1,8));ch=np.zeros((nu+1,8));hn=np.zeros((nu+1,4));hh=np.zeros((nu+1,4));dn=np.zeros((nu+1,2));dh=np.zeros((nu+1,2));first=np.zeros(nu+1);last=np.zeros(nu+1);lc=np.zeros(nu+1);ls=np.zeros(nu+1);run=np.zeros(nu+1);lasty=np.full(nu+1,-1.);history=np.zeros((nu+1,20));em2=np.full(nu+1,.63);em5=em2.copy();gaps=np.zeros(nu+1);gaps2=np.zeros(nu+1)
    start=0
    while start<len(order):
        end=start+1
        while end<len(order) and user[order[end]]==user[order[start]] and timestamp[order[end]]==timestamp[order[start]]:end+=1
        for pos in range(start,end):
            j=order[pos]
            u=user[j];t=timestamp[j];c=context[j];hour=c%4;day=c//4;n=count[u];rate=(hit[u]+6.3)/(n+10);r=result[j];r[0]=np.log1p(n);r[1]=(hit[u]+1.89)/(n+3);r[2]=rate;r[3]=(hit[u]+18.9)/(n+30);r[4]=rate*(1-rate)/(n+11)
            r[5]=(hh[u,hour]+2*rate)/(hn[u,hour]+2);r[6]=(hh[u,hour]+10*rate)/(hn[u,hour]+10);r[7]=(dh[u,day]+2*rate)/(dn[u,day]+2);r[8]=(dh[u,day]+10*rate)/(dn[u,day]+10);r[9]=(ch[u,c]+2*rate)/(cn[u,c]+2);r[10]=(ch[u,c]+10*rate)/(cn[u,c]+10);r[11]=lasty[u];r[12]=history[u,(n-2)%20] if n>=2 else -1
            for k,window in enumerate((3,5,10,20)):
                a=0.
                for z in range(min(n,window)):a+=history[u,(n-1-z)%20]
                r[13+k]=(a+2*.63)/(min(n,window)+2)
            r[17]=em2[u];r[18]=em5[u];r[19]=np.log1p(max(t-first[u],0)/3600) if n else -1;r[20]=np.log1p(max(t-last[u],0)/3600) if n else -1;r[21]=np.log1p(max(t-lc[u],0)/3600) if lc[u] else -1;r[22]=np.log1p(max(t-ls[u],0)/3600) if ls[u] else -1;r[23]=run[u];r[24]=gaps[u]/max(n-1,1);r[25]=np.sqrt(max(gaps2[u]/max(n-1,1)-r[24]**2,0));r[26]=(n+1)/max((t-1737331200)/86400,1);r[27]=(r[20]-r[24]) if n else 0;r[28]=np.log1p(max((t-1737331200)/86400,0));r[29]=np.log1p(hn[u,hour]);r[30]=np.log1p(dn[u,day]);r[31]=np.log1p(cn[u,c])
        for pos in range(start,end):
            j=order[pos];u=user[j];t=timestamp[j];c=context[j];hour=c%4;day=c//4;n=count[u]
            if target[j]<0:continue
            y=1.0 if target[j]<10 else 0.0
            if n==0:first[u]=t
            else:
                gap=np.log1p((t-last[u])/3600);gaps[u]+=gap;gaps2[u]+=gap*gap
            last[u]=t
            if y:lc[u]=t
            else:ls[u]=t
            run[u]=run[u]+1 if y==lasty[u] else 1;lasty[u]=y;history[u,n%20]=y;count[u]+=1;hit[u]+=y;cn[u,c]+=1;ch[u,c]+=y;hn[u,hour]+=1;hh[u,hour]+=y;dn[u,day]+=1;dh[u,day]+=y;em2[u]=.8*em2[u]+.2*y;em5[u]=.5*em5[u]+.5*y
        start=end
    return result

def main():
    d,e,a=load_data('v2/P3_Labs_model_and_code/cache');u=np.r_[d['user'],e['user']];t=np.r_[d['timestamp'],e['timestamp']];ctx=np.r_[d['context'],e['context']];y=np.r_[d['target'],np.full(len(e['user']),-1)];order=np.lexsort((t,u));x=rolling(u,t,ctx,y,order,a['n_users']);out=Path('v4/activity');out.mkdir(exist_ok=True);np.save(out/'train.npy',x[:len(d['user'])]);np.save(out/'evaluation.npy',x[len(d['user']):]);(out/'names.json').write_text(json.dumps(NAMES));print('Activity features',x.shape,flush=True)
    fi,vi=split_data(d,1737849600);target=(d['target'][vi]<10);base=np.load('v3/stack/cv_evidence.npz')['prediction'][-len(vi)*10:].reshape(-1,10).sum(1)
    print('baseline event auc',roc_auc_score(target,base),flush=True)
    for name in ['user_rate10','user_hour_rate2','user_context_rate2','last3_rate','last10_rate','ewma2','ewma5']:
        col=x[vi,NAMES.index(name)];print(name,roc_auc_score(target,col),'mix',roc_auc_score(target,.5*base+.5*col),flush=True)

if __name__=='__main__':main()
