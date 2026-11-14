"""Recreate the saved chronological Friday component comparison through V65."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from calibration_model import predict
from bounded_history import base_logits
from event_mass_correction import features,delta,transfer
from v30_tree import predict as tree_predict
from rebuild_benchmark import sha,write
from research_v43 import MASS,WEIGHTS

def main():
    p=argparse.ArgumentParser();p.add_argument('--benchmark',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2)
    folder=a.benchmark/'features/friday';al=dict(np.load(folder/'alignment.npz'));pos=al['first_position'];rows=(pos[:,None]*10+np.arange(10)).ravel()
    raw=np.load(folder/'raw.npy',mmap_mode='r')[rows];labels=al['labels'][rows];parts={};hashes={}
    expected=json.loads((ROOT/'results/v45_forward_diagnostic.json').read_text())
    for name in ['base','memory','small','slate']:
        path=ROOT/(f'work/recovered_controls/controls/{name}.pt' if name!='slate' else 'artifacts/v38/diagnostics/slate_head.pt')
        assert sha(path)==expected['model_sha256'][name];hashes[str(path.relative_to(ROOT))]=sha(path)
        model=torch.load(path,map_location='cpu',weights_only=False);assert model['train_events']==97717
        parts[name]=predict(model,raw[:,:344] if name=='base' else raw)
    path=ROOT/'artifacts/v38/diagnostics/stable_tree.txt';assert sha(path)==expected['tree_sha256'];hashes[str(path.relative_to(ROOT))]=sha(path)
    tree=lgb.Booster(model_file=str(path));ts=json.loads((path.parent/'stable_tree_spec.json').read_text());parts['tree']=tree_predict(tree,raw[:,:344],ts)
    pred=sum(w*parts[k] for k,w in WEIGHTS.items())/MASS
    assert abs(roc_auc_score(labels,pred)-expected['baseline_auc'])<1e-12
    for family in ['plain','slate']:
        path=ROOT/f'artifacts/v45/research/{family}/forward/model.pt';assert sha(path)==expected['new_model_sha256'][family]
        hashes[str(path.relative_to(ROOT))]=sha(path);parts['new_'+family]=predict(torch.load(path,map_location='cpu',weights_only=False),raw)
    pred+= (.109375*(parts['new_plain']-parts['small'])+.125*(parts['new_slate']-parts['slate']))/MASS
    assert abs(roc_auc_score(labels,pred)-expected['candidate_auc'])<1e-12
    for version,key in [(50,'narrow'),(63,'adapted')]:
        path=ROOT/f'artifacts/v{version}/research/forward/model.pt';hashes[str(path.relative_to(ROOT))]=sha(path)
        parts[key]=predict(torch.load(path,map_location='cpu',weights_only=False),raw)
    pred+=.0478515625*(2*parts['narrow']-parts['base']-parts['memory'])/MASS
    pred+=.0478515625*(2*parts['adapted']-parts['base']-parts['memory'])/MASS
    parts['v63']=pred.copy();assert abs(roc_auc_score(labels,pred)-.8250146662718386)<1e-12
    path=ROOT/'artifacts/v65/research/forward/model.pt';hashes[str(path.relative_to(ROOT))]=sha(path);model=torch.load(path,map_location='cpu',weights_only=False)
    utility=base_logits(model['base_spec'],raw);ef,names=features(raw,model['names'],utility);assert names==model['event_names'];parts['event_delta']=delta(model,ef)
    pred=transfer(pred,parts['event_delta'],1.);parts['v65']=pred
    auc=float(roc_auc_score(labels,pred));assert abs(auc-.8250189135906945)<1e-12
    np.savez_compressed(a.out/'components.npz',**parts)
    np.savez_compressed(a.out/'evidence.npz',indices=al['indices'][pos],labels=labels,users=al['user'][pos].repeat(10),baseline=parts['v63'],candidate=pred)
    report=dict(auc=auc,events=len(pos),model_sha256=hashes,source_sha256=sha(Path(__file__)),scope='Friday chronological normalized changed components; historical V17/V20 branches excluded. All heads fitted before Friday. No new fitting.')
    write(a.out/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
