"""Paired user-bootstrap assessment of a supplied, fixed candidate prediction."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss


def prepared_auc(labels,prediction):
    order=np.argsort(prediction,kind='stable');scores=prediction[order]
    ends=np.r_[np.flatnonzero(scores[1:]!=scores[:-1]),len(scores)-1]
    y=labels[order]
    def calculate(weights):
        w=weights[order];pos=np.cumsum(w*y)[ends];neg=np.cumsum(w*(1-y))[ends]
        if not pos[-1] or not neg[-1]:return float('nan')
        dp=np.diff(np.r_[0.,pos]);before=np.r_[0.,neg[:-1]]
        return float(np.dot(dp,(before+neg)/2)/(pos[-1]*neg[-1]))
    return calculate


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--out',required=True)
    p.add_argument('--candidate',default='v15_research');p.add_argument('--replicates',type=int,default=400)
    p.add_argument('--baseline',default='v13')
    p.add_argument('--seed',type=int,default=20261027)
    p.add_argument('--policy',type=Path,default=Path(__file__).resolve().parents[1]/'config/submission_policy.json')
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True);policy=json.loads(a.policy.read_text())
    d=np.load(a.evidence);y=d['labels'];base=d['baseline'];candidate=d['candidate'];users=d['users'];period=d['period']
    assert all(len(x)==len(y) for x in [base,candidate,users,period])
    assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1
    assert candidate.reshape(-1,10).sum(1).max()<=1+1e-6
    report={}
    for name,mask in [('combined',np.ones(len(y),bool)),('earlier',period==0),('later',period==1)]:
        report[name]={'rows':int(mask.sum())}
        for label,pred in [(a.baseline,base),(a.candidate,candidate)]:
            report[name][label]={'auc':float(roc_auc_score(y[mask],pred[mask])),
                                'logloss':float(log_loss(y[mask],pred[mask]))}
    _,group=np.unique(users,return_inverse=True);ng=int(group.max()+1)
    base_auc=prepared_auc(y,base);candidate_auc=prepared_auc(y,candidate)
    for name,fn in [(a.baseline,base_auc),(a.candidate,candidate_auc)]:
        assert abs(fn(np.ones(len(y)))-report['combined'][name]['auc'])<1e-12
    rng=np.random.default_rng(a.seed);deltas=[]
    for i in range(a.replicates):
        weight=rng.poisson(1,ng)[group].astype(np.float64)
        new=candidate_auc(weight);old=base_auc(weight)
        if i<2:
            assert abs(new-roc_auc_score(y,candidate,sample_weight=weight))<1e-12
            assert abs(old-roc_auc_score(y,base,sample_weight=weight))<1e-12
        deltas.append(new-old)
    gain=report['combined'][a.candidate]['auc']-report['combined'][a.baseline]['auc']
    interval=np.quantile(deltas,[.025,.975]).tolist()
    bootstrap={'auc_improvement':gain,'interval_95_percent':interval,'replicates':a.replicates,
               'seed':a.seed,'users':ng,'positive_replicates':int(np.sum(np.array(deltas)>0)),
               'method':'Paired Poisson bootstrap clustered by user across both dates.',
               'limitation':'Conditional on this selected candidate and reused development cohorts; excludes model-selection and distribution-shift uncertainty.'}
    gains={k:report[k][a.candidate]['auc']-report[k][a.baseline]['auc'] for k in report}
    date_pass=all(gains[k]>0 for k in ['earlier','later']);uncertainty_pass=interval[0]>0
    decision={'candidate':a.candidate,'incumbent':a.baseline,'local_gains':gains,'both_dates_improve':date_pass,
              'positive_bootstrap_lower_bound':uncertainty_pass,'remaining_slots_verified':False,
              'normal_quality_gate':gain>=policy['normal_minimum_local_gain'] and date_pass and uncertainty_pass,
              'last_two_quality_gate':gain>=policy['scarce_slots_minimum_local_gain'] and date_pass and uncertainty_pass,
              'official_score':None,'portal_uploads':0}
    for filename,obj in [('report.json',report),('paired_bootstrap.json',bootstrap),('decision.json',decision)]:
        (out/filename).write_text(json.dumps(obj,indent=2))
    print(json.dumps({'decision':decision,'bootstrap':bootstrap}),flush=True)


if __name__=='__main__':main()
