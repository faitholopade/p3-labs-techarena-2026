"""Small residual clock/position screens on saved V13 development predictions.

These are secondary development estimates, not nested independent validation.
"""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax,logsumexp
from sklearn.metrics import roc_auc_score


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True);ap.add_argument('--extension',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=np.load(Path(a.root)/'data/train.npz');e=np.load(Path(a.extension)/'validation/v13/evidence.npz')
    idx=e['indices'];fold=e['fold'].reshape(-1,10)[:,0];y=e['labels'].reshape(-1,10);base=e['v13'].reshape(-1,10)
    ts=data['timestamp'][idx];phase=(ts%86400)/86400
    x=np.c_[np.ones(len(idx)),np.sin(2*np.pi*phase),np.cos(2*np.pi*phase),np.sin(4*np.pi*phase),np.cos(4*np.pi*phase),
        *[((phase*4)%1-.5)*((phase*4).astype(int)==k) for k in range(4)]]
    utilities=np.log(base/np.maximum(1-base.sum(1,keepdims=True),1e-9))
    targets=np.where(y.sum(1)>0,y.argmax(1),10);all_results={};all_predictions={}
    for mode in ['position','time','joint']:
        use_position=mode!='time';use_time=mode!='position';dim=10*use_position+x.shape[1]*use_time
        pred=np.empty_like(base);fits=[]
        for f in range(3):
            tr=np.flatnonzero(fold!=f);va=np.flatnonzero(fold==f);xx=x[tr];oo=utilities[tr];yy=y[tr];tt=targets[tr]
            def objective(w):
                z=oo.copy();start=0
                if use_position:z+=w[:10];start=10
                if use_time:z+=(xx@w[start:])[:,None]
                zz=np.c_[z,np.zeros(len(tr))];prob=softmax(zz,axis=1)[:,:10];res=prob-yy
                value=(logsumexp(zz,axis=1)-zz[np.arange(len(tr)),tt]).sum()+500*np.sum(w*w)
                grads=[]
                if use_position:grads.append(res.sum(0))
                if use_time:grads.append(xx.T@res.sum(1))
                grad=np.concatenate(grads)+1000*w
                return value/len(tr),grad/len(tr)
            opt=minimize(objective,np.zeros(dim),jac=True,method='L-BFGS-B',options={'maxiter':100,'ftol':1e-12,'gtol':1e-8})
            if not opt.success:raise RuntimeError(str(opt.message))
            w=opt.x;z=utilities[va].copy();start=0
            if use_position:z+=w[:10];start=10
            if use_time:z+=(x[va]@w[start:])[:,None]
            pred[va]=softmax(np.c_[z,np.zeros(len(va))],axis=1)[:,:10]
            fits.append(w.tolist())
        report={'coefficients_by_fold':fits,'metrics':{}}
        for name,m in [('combined',np.ones(len(idx),bool)),('earlier',ts<1737849600),('later',ts>=1737849600)]:
            baseline_auc=roc_auc_score(y[m].ravel(),base[m].ravel());rr={}
            for w in [.5,1.]:
                auc=roc_auc_score(y[m].ravel(),((1-w)*base+w*pred)[m].ravel());rr[str(w)]={'auc':float(auc),'gain':float(auc-baseline_auc)}
            report['metrics'][name]=rr
        all_results[mode]=report;all_predictions[mode]=pred.ravel();print(mode,report['metrics'],flush=True)
    np.savez_compressed(out/'predictions.npz',**all_predictions)
    (out/'report.json').write_text(json.dumps({'method':'Regularized softmax offset correction, three user folds. Uses saved OOF base predictions; secondary stacking estimates are not fully nested.','results':all_results},indent=2))


if __name__=='__main__':main()
