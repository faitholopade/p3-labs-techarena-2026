"""Match the observed evaluation share of longer-gap first events using loss weights."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from daily_training import train
from calibration_model import predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','v20-evidence','v21-evidence','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--full',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    al=dict(np.load(a.features/'alignment.npz'));fs=json.loads((a.features/'spec.json').read_text());target=fs['evaluation_gap_ge_1day_fraction'];source=float(al['long_gap'].mean());factor=target*(1-source)/(source*(1-target));weights=np.where(al['long_gap'],factor,1.).astype(np.float32)
    assert abs(float(np.average(al['long_gap'],weights=weights))-target)<1e-6
    rule={'target_long_gap_fraction':target,'source_long_gap_fraction':source,'long_gap_relative_weight':factor,'target_uses_evaluation_timestamps_only':True,'limitation':'Observed 1-2-day training views represent all evaluation gaps of at least one day, including unsupported gaps beyond two days.'}
    (a.out/'weight_rule.json').write_text(json.dumps(rule,indent=2));np.save(a.out/'event_weights.npy',weights)
    raw=np.load(a.features/'raw.npy',mmap_mode='r');names=np.array(json.loads((a.features/'names.json').read_text()));config={'seed':20261104,'epochs':20,'width':32,'offset_name':'posterior1.0_mix1.0_utility','rank_weight':0.,'threads':a.threads,'architecture':'plain'}
    ds=[{'x':raw,'names':names,'labels':al['labels']}];started=time.time();records=[];prediction=np.empty(len(raw))
    for fold in ['full'] if a.full else range(3):
        mask=np.ones(len(al['indices']),bool) if fold=='full' else al['fold']!=fold;test=np.flatnonzero(~mask);assert not np.intersect1d(al['user'][mask],al['user'][test]).size
        model=train(ds,[mask],**config,sample_weights=[weights]);torch.save(model,a.out/('model.pt' if fold=='full' else f'fold_{fold}.pt'))
        if fold!='full':
            rows=(test[:,None]*10+np.arange(10)).ravel();prediction[rows]=predict(model,raw[rows])
        record={'fold':fold,'training_events':int(mask.sum()),'test_events':len(test),'users_disjoint':True,'seconds':time.time()-started};records.append(record);print(record,flush=True);del model;gc.collect()
    if a.full:
        (a.out/'spec.json').write_text(json.dumps({'config':config,'weight_rule':rule,'training_events':len(al['indices']),'feature_spec':fs},indent=2));return
    v20=dict(np.load(a.v20_evidence));v21=dict(np.load(a.v21_evidence));old=dict(np.load(a.features/'baseline_heads.npz'));r=(al['matched_position'][:,None]*10+np.arange(10)).ravel();long=np.repeat(al['long_gap'],10);small=(7*old['tree']+8*old['neural'])/15;results=[]
    for weight in [.03125,.0625,.125]:
        candidate=v20['candidate']+weight*(prediction[r]-old['neural'][r]);auc={};g20={};g21={}
        for name,mask in [('combined',np.ones(len(candidate),bool)),('earlier',v20['period']==0),('later',v20['period']==1)]:
            auc[name]=float(roc_auc_score(v20['labels'][mask],candidate[mask]));g20[name]=auc[name]-float(roc_auc_score(v20['labels'][mask],v20['candidate'][mask]));g21[name]=auc[name]-float(roc_auc_score(v21['labels'][mask],v21['candidate'][mask]))
        q=small+(weight/(15/64))*(prediction-old['neural']);gap=float(roc_auc_score(al['labels'][long],q[long])-roc_auc_score(al['labels'][long],small[long]));results.append({'replacement_weight':weight,'auc':auc,'gain_vs_v20':g20,'gain_vs_v21':g21,'long_gap_calibration_gain_vs_v20':gap})
    np.savez_compressed(a.out/'predictions.npz',prediction=prediction,matched_prediction=prediction[r],labels=al['labels'],indices=al['indices'],cutoffs=al['cutoffs'])
    report={'config':config,'weight_rule':rule,'folds':records,'results':results,'limits':'Adaptive experiment on reused dates. Covariate-only target weights do not establish absence of conditional distribution shift; longer evaluation gaps remain unsupported.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
