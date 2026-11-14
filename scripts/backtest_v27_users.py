"""Fresh weekday replication of the V17 users-only factor recipe (not full V20)."""
import argparse, json, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from common import load_data, subset, binary_labels, metrics, split_data
from backtest_v24_factor import fit_command, sha256, write_json
from v24_item_time import posterior_probability

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['root','fresh-core','out']:p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--cut',type=int,required=True);a=p.parse_args()
    a.root=a.root.resolve();a.fresh_core=a.fresh_core.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    original=json.loads((a.fresh_core/'plan.json').read_text())
    assert original['cut']==a.cut
    assert original['data_sha256']==sha256(a.root/'data/train.npz')
    core=a.fresh_core/'core/residual.pt';basis=a.fresh_core/'bases/exposure.npz'
    done=json.loads((a.fresh_core/'core/completed.json').read_text())
    assert sha256(core)==done['checkpoint_sha256']
    schedule=[
      {'name':'joint','script':'refine_factor_joint.py','epochs':24,'lr':.002,'seed':8819,'extra':[]},
      {'name':'prior','script':'refine_exposure_prior.py','epochs':16,'lr':.001,'seed':20261028,'extra':['--variance',.5,'--warm-weight',2,'--rank-weight',.5]},
      {'name':'antithetic','script':'refine_exposure_prior.py','epochs':16,'lr':.0007,'seed':20261030,'extra':['--variance',.5,'--warm-weight',2,'--rank-weight',.5,'--antithetic']},
      {'name':'users','script':'refine_exposure_prior.py','epochs':16,'lr':.003,'seed':20261031,'extra':['--variance',.5,'--warm-weight',2,'--rank-weight',.5,'--antithetic','--users-only']},
    ]
    plan={'cut':a.cut,'schedule':schedule,'core_sha256':sha256(core),'exposure_sha256':sha256(basis),
          'data_sha256':sha256(a.root/'data/train.npz'), 'posterior_samples':256,'threads':2,
          'source_sha256':{s:sha256(REPO/s) for s in ['scripts/backtest_v27_users.py','src/refine_factor_joint.py','src/refine_exposure_prior.py','src/posterior_inference.py','src/residual_model.py']},
          'scope':'V17 users-only factor family, newly fitted before each weekday. No future checkpoint or calibration head. Not the full V20 ensemble.'}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen plan changed.')
    write_json(path,plan);started=time.time();previous=core
    for step in schedule:
        args=[REPO/'src'/step['script'],'--root',a.root,'--init',previous,'--exposure',basis,
              '--cut',a.cut,'--out',a.out/step['name'],'--pool-cache',a.fresh_core/'pools',
              '--threads',2,'--epochs',step['epochs'],'--lr',step['lr'],'--seed',step['seed']]+step['extra']
        previous=fit_command(args,a.out/step['name'],'residual.pt')
    data,_,audit=load_data(a.root/'data');_,ix=split_data(data,a.cut);ix=ix[data['timestamp'][ix]<a.cut+86400]
    query=subset(data,ix);labels=binary_labels(query);query['target']=np.full(len(ix),-1)
    assert np.all((query['hist']==0)|(query['hist_ts']<a.cut))
    ck=torch.load(previous,map_location='cpu',weights_only=False);assert ck['cut']==a.cut
    probability=posterior_probability(ck['state'],query,samples=256)
    np.savez_compressed(a.out/'predictions.npz',indices=ix,labels=labels,user=query['user'],timestamp=query['timestamp'],probability=probability)
    report={'plan':plan,'events':len(ix),'metrics':metrics(labels,probability),'factor_sha256':sha256(previous),
            'seconds':time.time()-started,'limits':'Raw factor score for this fresh weekday, not a V20 improvement estimate.'}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
