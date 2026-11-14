"""Score fixed temporal user fits using shared posterior draws and matched V20 rows."""
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from posterior_inference import features as posterior_features


def corrected(prob,delta,weight,conditional):
    p=np.asarray(prob).reshape(-1,10);d=np.asarray(delta).reshape(-1,10)
    if conditional:return (p.sum(1,keepdims=True)*softmax(np.log(np.maximum(p,1e-12))+weight*d,1)).ravel()
    utility=np.log(np.maximum(p,1e-12)/np.maximum(1-p.sum(1,keepdims=True),1e-12))
    return softmax(np.c_[utility+weight*d,np.zeros(len(p))],1)[:,:10].ravel()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['features','experiments','incumbent','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--name',required=True);p.add_argument('--period',choices=['earlier','later','both'],default='both');a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    al=dict(np.load(a.features/'alignment.npz'));raw=np.load(a.features/'raw.npy',mmap_mode='r');names=json.loads((a.features/'names.json').read_text());col=names.index('posterior1.0_mix1.0_utility')
    for period in ['earlier','later'] if a.period=='both' else [a.period]:
        ck=torch.load(a.experiments/(period+'_'+a.name)/'model.pt',map_location='cpu',weights_only=False);d=dict(np.load(a.features/(period+'_query.npz')))
        assert np.all(d['target']==-1) and np.all((d['hist']==0)|(d['hist_ts']<ck['cut']))
        chol=None
        if 'correlation_raw' in ck:
            from refine_user_correlation import lower_factor
            chol,_=lower_factor(ck['correlation_raw'])
        start=time.time();value,pnames=posterior_features(ck['state'],d,samples=256,user_cholesky=chol);position=np.flatnonzero(al['period']==(['earlier','later'].index(period)));rows=(position[:,None]*10+np.arange(10)).ravel()
        delta=value[:,pnames.index('posterior1.0_mix1.0_utility')]-raw[rows,col]
        np.savez_compressed(a.out/(period+'.npz'),delta=delta,position=position)
        print({'period':period,'seconds':time.time()-start,'mean_absolute_delta':float(np.abs(delta).mean())},flush=True)
    if a.period!='both':return
    delta=np.empty(len(raw))
    for period in ['earlier','later']:
        d=np.load(a.out/(period+'.npz'));rr=(d['position'][:,None]*10+np.arange(10)).ravel();delta[rr]=d['delta']
    ev=dict(np.load(a.incumbent));r=(al['matched_position'][:,None]*10+np.arange(10)).ravel();assert np.array_equal(ev['labels'],al['labels'][r]);y=ev['labels']
    small=np.load(a.features/'baseline_heads.npz');q=(7*small['tree']+8*small['neural'])/15;long=np.repeat(al['long_gap'],10);report=[]
    for conditional in [False,True]:
        for weight in [.125,.25,.5]:
            candidate=corrected(ev['candidate'],delta[r],weight,conditional);gains={};auc={}
            for name,m in [('combined',np.ones(len(y),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
                auc[name]=float(roc_auc_score(y[m],candidate[m]));gains[name]=auc[name]-float(roc_auc_score(y[m],ev['candidate'][m]))
            qnew=corrected(q,delta,weight,conditional);gap=float(roc_auc_score(al['labels'][long],qnew[long])-roc_auc_score(al['labels'][long],q[long]))
            report.append({'conditional':conditional,'weight':weight,'auc':auc,'gain_vs_v20':gains,'long_gap_calibration_gain':gap})
    np.savez_compressed(a.out/'correction.npz',delta=delta,matched_delta=delta[r]);(a.out/'report.json').write_text(json.dumps({'name':a.name,'results':report,'limits':'Reused validation dates; long-gap calibration diagnostic is not a full-model or independent comparison.'},indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
