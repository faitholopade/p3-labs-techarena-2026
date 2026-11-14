"""Train only Thursday; compare memory and seed averaging on first Friday queries."""
import argparse,gc,json,sys,time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,log_loss
from daily_training import train
from calibration_model import predict
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    folders=[a.features/'thursday',a.features/'friday'];align=[dict(np.load(f/'alignment.npz')) for f in folders]
    raw=[np.load(f/'raw.npy',mmap_mode='r') for f in folders];names=[json.loads((f/'names.json').read_text()) for f in folders]
    assert names[0]==names[1]
    assert align[0]['timestamp'].max()<align[1]['timestamp'].min()
    pos=align[1]['first_position'];rows=(pos[:,None]*10+np.arange(10)).ravel();y=align[1]['labels'][rows]
    configs=[('base',20261104),('base',20261029),('base',20261030),('memory',20261029),('memory',20261030)]
    plan={'training':'All Thursday events only','validation':'First query per user on Friday','configurations':[list(c) for c in configs],
          'epochs':20,'width':32,'rank_weight':.5,'architecture':'plain','threads':2,'selection':'Fixed ensembles; no tuning to Friday.',
          'source_sha256':sha256(Path(__file__)),'feature_reports':[sha256(f/'report.json') for f in folders],
          'v25_confirmation':'Use the 0.125 blend weight selected on weekend data; fixed before Friday results. Also retain full-head memory versus control comparison.',
          'limits':'Newly fitted heads on established factor recipe; not the complete V20 weekday baseline. Friday has been inspected in earlier research.'}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen plan changed.')
    write_json(path,plan);parts={};records=[];started=time.time()
    for variant,seed in configs:
        columns=344 if variant=='base' else len(names[0]);modelpath=a.out/f'{variant}_{seed}.pt';done=a.out/f'{variant}_{seed}_completed.json'
        if done.exists():
            record=json.loads(done.read_text());assert sha256(modelpath)==record['model_sha256'];model=torch.load(modelpath,map_location='cpu',weights_only=False)
        else:
            if modelpath.exists():raise ValueError('Unverified interrupted model.')
            model=train([{'x':raw[0][:,:columns],'labels':align[0]['labels'],'names':np.array(names[0][:columns])}],
                        [np.ones(len(align[0]['indices']),bool)],seed=seed,epochs=20,width=32,rank_weight=.5,threads=2,
                        architecture='plain',offset_name='posterior1.0_mix1.0_utility')
            torch.save(model,modelpath);record={'variant':variant,'seed':seed,'model_sha256':sha256(modelpath),'seconds':time.time()-started};write_json(done,record)
        parts[f'{variant}_{seed}']=predict(model,raw[1][rows,:columns]);records.append(record);del model;gc.collect();print(record,flush=True)
    baseline=parts['base_20261104'];control=(parts['base_20261029']+parts['base_20261030'])/2
    memory=(parts['memory_20261029']+parts['memory_20261030'])/2;seedmean=(baseline+2*control)/3
    result={}
    for version,parent,candidate in [('v25_raw_head',control,memory),('v25',control,.875*control+.125*memory),('v26',baseline,seedmean)]:
        result[version]={'baseline_auc':float(roc_auc_score(y,parent)),'candidate_auc':float(roc_auc_score(y,candidate)),
                         'gain':float(roc_auc_score(y,candidate)-roc_auc_score(y,parent)),
                         'logloss_change':float(log_loss(y,candidate)-log_loss(y,parent))}
        np.savez_compressed(a.out/f'{version}_evidence.npz',indices=align[1]['indices'][pos],labels=y,baseline=parent,candidate=candidate,users=np.repeat(align[1]['user'][pos],10))
    np.savez_compressed(a.out/'all_predictions.npz',**parts)
    report={'plan':plan,'models':records,'results':result,'events':len(pos),'seconds':time.time()-started,'portal_uploads':0}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
