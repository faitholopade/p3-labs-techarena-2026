"""User-held-out calibration on first-event queries with a long-gap diagnostic."""
import argparse,gc,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sklearn.metrics import roc_auc_score
from daily_training import train
from calibration_model import predict
from flexible_tree import fit as tree_fit,predict as tree_predict


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['features','incumbent','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--kind',choices=['tree','neural'],default='neural');p.add_argument('--threads',type=int,default=2)
    p.add_argument('--architecture',choices=['plain','set'],default='plain');p.add_argument('--rank-weight',type=float,default=.5)
    p.add_argument('--seed',type=int,default=20261104)
    p.add_argument('--full',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);start=time.time()
    raw=np.load(a.features/'raw.npy',mmap_mode='r');al=dict(np.load(a.features/'alignment.npz'))
    names=json.loads((a.features/'names.json').read_text());ev=dict(np.load(a.incumbent));old=dict(np.load(a.features/'baseline_heads.npz'))
    assert np.array_equal(al['matched_indices'],ev['indices']); position=al['matched_position'];n=len(al['indices'])
    if a.kind=='neural':config={'epochs':20,'width':32,'rank_weight':a.rank_weight,'seed':a.seed,'threads':a.threads,'architecture':a.architecture,'offset_name':'posterior1.0_mix1.0_utility'}
    else:config={'mode':'event','rounds':700,'leaves':7,'seed':20261103,'threads':a.threads,'min_leaf':2000,'l2':100.}
    def fit(mask,path):
        if a.kind=='neural':
            model=train([{'x':raw,'labels':al['labels'],'names':np.array(names)}],[mask],**config);torch.save(model,path);return model,None
        rows=(np.flatnonzero(mask)[:,None]*10+np.arange(10)).ravel(); model,spec=tree_fit(raw[rows],al['labels'][rows],names,config);model.save_model(str(path));return model,spec
    if a.full:
        model,spec=fit(np.ones(n,bool),a.out/('model.pt' if a.kind=='neural' else 'model.txt'))
        (a.out/'spec.json').write_text(json.dumps({'kind':a.kind,'config':config,'tree_spec':spec,'feature_spec':json.loads((a.features/'spec.json').read_text()),'training_events':n,'training_rows':n*10},indent=2));return
    prediction=np.empty(n*10);records=[]
    for fold in range(3):
        training=al['fold']!=fold;test=np.flatnonzero(~training)
        assert not np.intersect1d(al['user'][training],al['user'][test]).size
        model,spec=fit(training,a.out/(f'fold_{fold}.'+('pt' if a.kind=='neural' else 'txt')))
        rows=(test[:,None]*10+np.arange(10)).ravel()
        prediction[rows]=predict(model,raw[rows]) if a.kind=='neural' else tree_predict(model,raw[rows],spec)
        rec={'fold':fold,'training_events':int(training.sum()),'test_events':len(test),'users_disjoint':True,'seconds':time.time()-start};records.append(rec);print(rec,flush=True);del model;gc.collect()
    r=(position[:,None]*10+np.arange(10)).ravel();pred=prediction[r]
    np.savez_compressed(a.out/'predictions.npz',prediction=prediction,matched_prediction=pred,labels=al['labels'],indices=al['indices'],cutoffs=al['cutoffs'])
    key='neural' if a.kind=='neural' else 'tree';weight=.125 if a.kind=='neural' else .109375
    long=np.repeat(al['long_gap'],10);small_base=(7*old['tree']+8*old['neural'])/15; results=[]
    for fraction in [.5,1.]:
        candidate=ev['candidate']+fraction*weight*(pred-old[key][r]);auc={};gain={}
        for label,mask in [('combined',np.ones(len(candidate),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            auc[label]=float(roc_auc_score(ev['labels'][mask],candidate[mask]));gain[label]=auc[label]-float(roc_auc_score(ev['labels'][mask],ev['candidate'][mask]))
        small_new=small_base+fraction*(8/15 if key=='neural' else 7/15)*(prediction-old[key])
        gap_gain=float(roc_auc_score(al['labels'][long],small_new[long])-roc_auc_score(al['labels'][long],small_base[long]))
        results.append({'fraction_replaced':fraction,'effective_weight':fraction*weight,'auc':auc,'gain':gain,'long_gap_calibration_subensemble_gain':gap_gain})
    report={'config':config,'folds':records,'results':results,'base_version':'v20','kind':a.kind,
            'limits':'Repeated development cohorts; long-gap diagnostic compares only a calibration subensemble and reuses outcomes. Not full-incumbent or independent-holdout evidence.'}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(results,indent=2),flush=True)


if __name__=='__main__':main()
