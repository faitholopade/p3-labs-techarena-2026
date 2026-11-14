"""Past-only intercept posterior: evaluate fixed corrections against V20 and V21."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score
from common import load_data,subset
from residual_model import load
from intercept_posterior import fit,predict,log_mass


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['root','features','v17-extension','v20-evidence','v21-evidence','v21-head','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(a.threads)
    data,_,_=load_data(a.root/'data');al=dict(np.load(a.features/'alignment.npz'));result={str(c):np.empty(len(al['indices'])) for c in [.25,1.]};point=np.empty(len(al['indices']));records=[];started=time.time()
    for period,cut in [('earlier',1737763200),('later',1737849600)]:
        rows=np.flatnonzero(al['cutoffs']==cut);query=subset(data,al['indices'][rows]);query['target']=np.full(len(rows),-1,dtype=np.int32);past=subset(data,np.flatnonzero(data['timestamp']<cut))
        factor=a.v17_extension/f'validation/factors/{period}_factor_users_only.pt';ck=torch.load(factor,map_location='cpu',weights_only=False);assert ck['cut']==cut
        net=load(factor);train_mass=log_mass(net,past);query_mass=log_mass(net,query);outside=net.outside.weight.detach().numpy()[:,0];point[rows]=expit(query_mass+outside[query['user']]);prior=ck['state']['outside_prior'].numpy()
        for precision in [.25,1.]:
            posterior=fit(past['user'],train_mass,past['target']<10,prior,precision=precision,nodes=31);rate,variance=predict(posterior,query['user'],query_mass);result[str(precision)][rows]=rate
            np.savez_compressed(a.out/f'{period}_precision{precision}.npz',**posterior,query_rate=rate,query_variance=variance,query_mass=query_mass,query_users=query['user'])
            record={'period':period,'cutoff':cut,'prior_precision':precision,'past_events':len(past['user']),'query_events':len(rows),'mean_posterior_sd':float(np.sqrt(posterior['variance'][query['user']]).mean()),'mean_absolute_intercept_shift':float(np.mean(abs(posterior['mean'][query['user']]-outside[query['user']]))),'seconds':time.time()-started};records.append(record);print(record,flush=True)
        del past,query,net,ck;gc.collect()
    np.savez_compressed(a.out/'predictions.npz',point_event_rate=point,**{'posterior_'+key.replace('.','_'):value for key,value in result.items()})
    ev20=dict(np.load(a.v20_evidence));ev21=dict(np.load(a.v21_evidence));positions=al['matched_position'];old=dict(np.load(a.features/'baseline_heads.npz'));new21=np.load(a.v21_head)['prediction']
    small20=(7*old['tree']+8*old['neural'])/15;small21=small20+(4/15)*(new21-old['neural']);long=np.repeat(al['long_gap'],10);grid=[]
    for parent,base,small in [('v20',ev20['candidate'],small20),('v21',ev21['candidate'],small21)]:
        b=base.reshape(-1,10);rate=b.sum(1);small=small.reshape(-1,10);small_rate=small.sum(1)
        for precision,q in result.items():
            for mode in ['blend_event_rate','relative_logit_correction']:
                for strength in [.0625,.125,.25]:
                    if mode=='blend_event_rate':nr=(1-strength)*rate+strength*q[positions];ns=(1-strength)*small_rate+strength*q
                    else:
                        delta=logit(np.clip(q,1e-9,1-1e-9))-logit(np.clip(point,1e-9,1-1e-9));nr=expit(logit(np.clip(rate,1e-9,1-1e-9))+strength*delta[positions]);ns=expit(logit(np.clip(small_rate,1e-9,1-1e-9))+strength*delta)
                    candidate=(b*(nr/np.maximum(rate,1e-12))[:,None]).ravel();small_new=(small*(ns/np.maximum(small_rate,1e-12))[:,None]).ravel();auc={};g20={};g21={}
                    for name,mask in [('combined',np.ones(len(base),bool)),('earlier',ev20['period']==0),('later',ev20['period']==1)]:
                        auc[name]=float(roc_auc_score(ev20['labels'][mask],candidate[mask]));g20[name]=auc[name]-float(roc_auc_score(ev20['labels'][mask],ev20['candidate'][mask]));g21[name]=auc[name]-float(roc_auc_score(ev21['labels'][mask],ev21['candidate'][mask]))
                    gap=float(roc_auc_score(al['labels'][long],small_new[long])-roc_auc_score(al['labels'][long],small.ravel()[long]));grid.append({'parent':parent,'prior_precision':precision,'mode':mode,'strength':strength,'auc':auc,'gain_vs_v20':g20,'gain_vs_v21':g21,'long_gap_calibration_gain_vs_parent':gap})
    report={'records':records,'grid':grid,'limits':'Exploratory fixed-factor conditional posterior; latent item/user parameters are held at their fitted values. No validation/query outcome enters posterior fitting. Date cohorts and outcome-reusing long-gap diagnostic remain reused development data.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(sorted(grid,key=lambda x:x['gain_vs_v20']['combined'],reverse=True)[:5],indent=2),flush=True)


if __name__=='__main__':main()
