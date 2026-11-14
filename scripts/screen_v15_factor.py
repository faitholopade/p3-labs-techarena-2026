"""Score a changed chronological factor fit through frozen user-excluded heads."""
import sys,argparse,json,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from sklearn.metrics import roc_auc_score,log_loss
from factor_ensemble import update_features
from sequence_calibration import history_pairs,predict as sequence_predict
from calibration_model import predict as plain_predict
from posterior_inference import features as posterior_features


def main():
    p=argparse.ArgumentParser();p.add_argument('--period',choices=['earlier','later'],required=True)
    p.add_argument('--factor',required=True);p.add_argument('--extension',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True)
    p.add_argument('--correlation');p.add_argument('--correlation-weight',type=float,default=1.)
    p.add_argument('--posterior-samples',type=int,default=64)
    p.add_argument('--incumbent-evidence');p.add_argument('--incumbent-name',default='v13')
    p.add_argument('--v15-evidence',help='For V16, also compare an exact replacement of its newest factor branch.')
    p.add_argument('--parent-screen',help='Matched stored sequence predictions of the factor being replaced.')
    p.add_argument('--replacement-share',type=float,help='Its exact probability share in the incumbent.')
    p.add_argument('--exact-direction',action='store_true',help='Use exact isotropic projected-normal mean directions for this branch only.')
    a=p.parse_args();cache=Path(a.cache);extension=Path(a.extension);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if not (cache/'parity.json').exists():raise ValueError('Complete the baseline parity check before screening.')
    parity=json.loads((cache/'parity.json').read_text())
    if parity['sequence_candidate_max_abs']>=1e-6:raise ValueError('Baseline parity failed.')
    raw=np.load(cache/f'{a.period}_raw.npy').copy();query=dict(np.load(cache/f'{a.period}_query.npz'))
    align=np.load(cache/f'{a.period}_alignment.npz');selection=align['selection'];fold=align['fold'];rows=(selection[:,None]*10+np.arange(10)).ravel()
    ev=np.load(extension/'validation/v13/evidence.npz');baseline=ev['v13'][rows];labels=ev['labels'][rows];v12=ev['v12'][rows]
    if a.incumbent_evidence:
        incumbent=np.load(a.incumbent_evidence)
        assert np.array_equal(incumbent['indices'],ev['indices']) and np.array_equal(incumbent['labels'],ev['labels'])
        baseline=incumbent['candidate'][rows]
    legacy=np.load(cache/'legacy.npy')[rows]
    update_features(raw,query,a.factor,posterior_samples=a.posterior_samples,exact_direction=a.exact_direction)
    state=torch.load(a.factor,map_location='cpu',weights_only=False)['state'];pairs,mask=history_pairs(state,query,exact_direction=a.exact_direction)
    if a.correlation:
        corr=torch.tensor(np.load(a.correlation));rho=a.correlation_weight
        if not 0<=rho<=1:raise ValueError('Correlation weight must be in [0,1].')
        expected=8+state['exposure_user.weight'].shape[1]
        if corr.shape[1:]!=(expected,expected):raise ValueError('Posterior correlation dimension mismatch.')
        chol=torch.linalg.cholesky(rho*corr+(1-rho)*torch.eye(expected)[None])
        extra,extra_names=posterior_features(state,query,samples=a.posterior_samples,user_cholesky=chol)
        names=json.loads((cache/f'{a.period}_names.json').read_text())
        raw[:,[names.index(n) for n in extra_names]]=extra
    sequence=np.empty(len(raw));plain=np.empty(len(raw))
    for f in range(3):
        events=np.flatnonzero(fold==f);rr=(events[:,None]*10+np.arange(10)).ravel();ss=[];pp=[]
        for seed in [731,932,1417]:
            ck=torch.load(extension/'validation/v13/fold_models'/f'head_{f}_{seed}.pt',map_location='cpu',weights_only=False)
            ss.append(sequence_predict(ck,raw[rr],pairs[events],mask[events]));pp.append(plain_predict(ck['base_spec'],raw[rr]))
        sequence[rr]=np.mean(ss,axis=0);plain[rr]=np.mean(pp,axis=0)
    candidates={'sequence_replacement':.5*v12+.1875*legacy+.3125*sequence,
                'sequence_branch':.375*legacy+.625*sequence,
                'plain_sequence_branch':.375*legacy+.3125*plain+.3125*sequence}
    if a.v15_evidence:
        if a.incumbent_name!='v16':raise ValueError('The exact factor replacement formula requires a V16 incumbent.')
        parent=np.load(a.v15_evidence)
        assert np.array_equal(parent['indices'],ev['indices']) and np.array_equal(parent['labels'],ev['labels'])
        candidates['incumbent_factor_replacement']=.75*parent['candidate'][rows]+.09375*legacy+.15625*sequence
    if a.parent_screen:
        if a.v15_evidence or a.replacement_share is None or not 0<a.replacement_share<=1:raise ValueError('Specify one exact replacement with a valid probability share.')
        parent=np.load(a.parent_screen)
        assert np.array_equal(parent['selection'],selection) and np.array_equal(parent['labels'],labels)
        candidates['incumbent_factor_replacement']=baseline+a.replacement_share*(sequence-parent['sequence'])
    report={'period':a.period,'factor':a.factor,'posterior_samples':a.posterior_samples,'exact_direction':a.exact_direction,'baseline_version':a.incumbent_name,'baseline_auc':float(roc_auc_score(labels,baseline)),'candidates':{}}
    np.savez_compressed(out/'predictions.npz',selection=selection,labels=labels,baseline=baseline,sequence=sequence,plain=plain,**candidates)
    for name,pred in candidates.items():
        assert np.isfinite(pred).all() and pred.min()>=0 and pred.max()<=1 and pred.reshape(-1,10).sum(1).max()<=1+1e-6
        results={}
        for weight in [.25,.5,.75,1.]:
            blended=(1-weight)*baseline+weight*pred
            auc=float(roc_auc_score(labels,blended))
            results[str(weight)]={'auc':auc,'gain':auc-report['baseline_auc']}
        report['candidates'][name]=results
    (out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':main()
