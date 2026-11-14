"""Training-only rate estimates from observed exposures and censored click histories."""
import numpy as np

NAMES=['log_event_count','log_observed_click_count','log_history_count','log_history_window',
       'log_event_rate','log_survival_event_rate','log_history_click_rate',
       'log_click_event_ratio','log_click_survival_ratio','history_censored',
       'log_query_delay','log_click_age','log_history_span','mean_log_click_gap',
       'std_log_click_gap','recent_to_full_rate_1','recent_to_full_rate_3',
       'recent_to_full_rate_7','observed_vs_history_rate','smoothed_train_click_logit']


def features(fit, query, cutoff, n_users):
    counts=np.bincount(fit['user'],minlength=n_users+1)
    clicked=np.bincount(fit['user'],weights=fit['target']<10,minlength=n_users+1)
    user=query['user'];t=query['timestamp'];h=query['hist'];ht=query['hist_ts'];mask=h>0
    n=mask.sum(1);censored=n==h.shape[1]
    assert np.all(ht[mask]<np.broadcast_to(t[:,None],mask.shape)[mask])
    warm_start=int(fit['hist_ts'][fit['hist']>0].min())
    train_start=int(fit['timestamp'].min());duration=(cutoff-train_start)/86400
    oldest=np.where(mask,ht,t[:,None]).min(1);last=np.where(mask,ht,warm_start).max(1)
    window=np.maximum((t-np.where(censored,oldest,warm_start))/86400,1e-3)
    click_rate=(n-censored+1)/(window+2)
    event_rate=(counts[user]+2)/(duration+2)
    wait=np.maximum((t-cutoff)/86400,0)
    survival_rate=(counts[user]+3)/(duration+2+wait)
    gaps=np.maximum(np.diff(ht,axis=1)/86400,0);gm=mask[:,1:]&mask[:,:-1]
    lg=np.log1p(gaps);ng=np.maximum(gm.sum(1),1)
    mean=(lg*gm).sum(1)/ng;std=np.sqrt(np.maximum((lg*lg*gm).sum(1)/ng-mean*mean,0))
    p=(clicked[user]+3.15)/(counts[user]+5)
    columns=[np.log1p(counts[user]),np.log1p(clicked[user]),np.log1p(n),np.log1p(window),
             np.log(event_rate),np.log(survival_rate),np.log(click_rate),
             np.log(click_rate/event_rate),np.log(click_rate/survival_rate),censored,
             np.log1p(wait),np.log1p(np.maximum((t-last)/86400,0)),
             np.log1p(np.maximum((last-oldest)/86400,0)),mean,std]
    age=(t[:,None]-ht)/86400
    for days in [1,3,7]:
        recent=((age<days)&mask).sum(1)
        columns.append(np.log((recent+1)/(days+2)/click_rate))
    columns.extend([np.log((clicked[user]+1)/(duration+2)/click_rate),np.log(p/(1-p))])
    out=np.stack(columns,axis=1).astype(np.float32)
    assert np.isfinite(out).all() and out.shape[1]==len(NAMES)
    return out
