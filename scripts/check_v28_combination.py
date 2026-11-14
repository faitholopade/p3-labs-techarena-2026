"""One fixed 6.25% memory addition to selected V27; no weight search."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v27','memory','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--plan-only',action='store_true');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plan={'version':'v28','memory_weight':.0625,'formula':'.9375*selected_v27 + .0625*mean_memory_head',
          'v27_sha256':sha256(a.v27),'memory_sha256':sha256(a.memory),'source_sha256':sha256(Path(__file__)),
          'advance':'At least +0.000005 combined AUC versus V27, both dates improve versus V27. No additional weight search.',
          'selection_limit':'Proposed after V25 and V27 results, fixed before this combination is evaluated.'}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Frozen combination changed.')
    write_json(path,plan)
    if a.plan_only:return
    z=dict(np.load(a.v27));memory=np.load(a.memory)['memory'];candidate=.9375*z['candidate']+.0625*memory
    assert memory.shape==z['candidate'].shape and np.isfinite(candidate).all()
    masks={'combined':np.ones(len(candidate),bool),'earlier':z['period']==0,'later':z['period']==1}
    gains={baseline:{name:float(roc_auc_score(z['labels'][m],candidate[m])-roc_auc_score(z['labels'][m],z[field][m])) for name,m in masks.items()} for baseline,field in [('v20','baseline'),('v27','candidate')]}
    advance=gains['v27']['combined']>=.000005 and gains['v27']['earlier']>0 and gains['v27']['later']>0
    report={'plan':plan,'gains':gains,'advance':advance,'global_logloss_change_vs_v27':float(log_loss(z['labels'],candidate)-log_loss(z['labels'],z['candidate'])),'portal_uploads':0}
    np.savez_compressed(a.out/'evidence.npz',**{k:v for k,v in z.items() if k!='candidate'},candidate=candidate)
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
