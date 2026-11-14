"""Read-only train/eval feature audit; never uses evaluation targets."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.stats import chisquare, chi2_contingency

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--data',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args();D=a.data
a.out.parent.mkdir(parents=True,exist_ok=True)
t=dict(np.load(D/'train.npz')); e=dict(np.load(D/'evaluation.npz'))
NI=50001; NU=54457
def quant(x):
    return dict(zip(['min','p10','p25','median','p75','p90','max'],map(float,np.quantile(x,[0,.1,.25,.5,.75,.9,1]))))
def subset(d,ix): return {k:v[ix] for k,v in d.items() if k!='target'}
def cohort(q,cut):
    p=t['timestamp']<cut; u=t['user'][p]; c=t['context'][p]; y=t['target'][p]<10
    n=np.bincount(u,minlength=NU); hit=np.bincount(u,weights=y,minlength=NU)
    ctxn=np.bincount(u*8+c,minlength=NU*8).reshape(NU,8)
    dn=np.bincount(u*2+c//4,minlength=NU*2).reshape(NU,2)
    ci=np.bincount((t['items'][p]*8+c[:,None]).ravel(),minlength=NI*8).reshape(NI,8)
    qu=q['user']; qc=q['context']; h=q['hist']>0; age=(q['timestamp'][:,None]-q['hist_ts'])/86400
    return {'events':len(qu),'ctx_counts':np.bincount(qc,minlength=8).tolist(),
      'delay_days':quant((q['timestamp']-cut)/86400),
      'user_prior_events':quant(n[qu]),'cold_user_share':float(np.mean(n[qu]==0)),
      'user_prior_context_events':quant(ctxn[qu,qc]),'user_prior_daytype_events':quant(dn[qu,qc//4]),
      'item_prior_context_impressions':quant(ci[q['items'],qc[:,None]].ravel()),
      'user_smoothed_rate':quant((hit[qu]+6.3)/(n[qu]+10)),
      'history_length':quant(h.sum(1)),
      'history_oldest_days':quant(np.where(h,age,0).max(1)),
      'history_latest_days':quant(np.where(h,age,np.inf).min(1)[h.any(1)]),
      'supplied_hist_after_cut':int(np.count_nonzero(h&(q['hist_ts']>=cut))),
      'query_hist_not_strictly_past':int(np.count_nonzero(h&(q['hist_ts']>=q['timestamp'][:,None])))}

out={'notes':['Evaluation targets ignored.','All cohort features use outcomes strictly before cutoff.']}
position=np.bincount(t['target'],minlength=11)
out['position']={'counts':position.tolist(),'clicked_position_uniform_chisquare':float(chisquare(position[:10]).statistic),'pvalue':float(chisquare(position[:10]).pvalue)}
out['train_day_context']=[]
for day in np.unique(t['timestamp']//86400):
    ix=t['timestamp']//86400==day
    for c in np.unique(t['context'][ix]):
        m=ix&(t['context']==c); yy=t['target'][m]<10
        out['train_day_context'].append({'day':int(day),'context':int(c),'events':int(m.sum()),'click_rate':float(yy.mean())})
out['cohorts']={}
for cut in (1737590400,1737676800,1737763200,1737849600):
    ix=np.flatnonzero((t['timestamp']>=cut)&(t['timestamp']<cut+86400))
    _,pos=np.unique(t['user'][ix],return_index=True);ix=ix[np.sort(pos)]
    z=cohort(subset(t,ix),cut);z['click_rate']=float((t['target'][ix]<10).mean());out['cohorts'][str(cut)]=z
out['cohorts']['eval']=cohort(e,1737936000)
a.out.write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
