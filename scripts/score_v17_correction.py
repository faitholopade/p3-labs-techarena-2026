"""Assess a bounded utility correction from a changed V16 factor branch."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.metrics import roc_auc_score


def utility(p):
    p=np.clip(p,1e-9,1-1e-9);outside=np.maximum(1-p.sum(1,keepdims=True),1e-9)
    return np.log(p/outside)


def corrected(base,delta,weight,conditional=False):
    if conditional:return base.sum(1,keepdims=True)*softmax(np.log(np.clip(base,1e-9,1))+weight*delta,1)
    return softmax(np.c_[utility(base)+weight*delta,np.zeros(len(base))],1)[:,:10]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v15-evidence','v16-evidence','legacy','screen','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();parent=np.load(a.v15_evidence);inc=np.load(a.v16_evidence);screen=np.load(a.screen)
    select=screen['selection'];rows=(select[:,None]*10+np.arange(10)).ravel();base=inc['candidate'][rows].reshape(-1,10)
    oldseq=((inc['candidate']-.75*parent['candidate']-.09375*np.load(a.legacy))/.15625)[rows].reshape(-1,10)
    newseq=screen['sequence'].reshape(-1,10);delta=utility(newseq)-utility(oldseq)
    # Float32 head averaging is recovered from the incumbent mixture to roundoff.
    changed=np.max(np.abs(newseq-oldseq),1)>1e-7;delta[~changed]=0
    y=inc['labels'][rows];auc=roc_auc_score(y,base.ravel());results=[]
    for conditional in [False,True]:
        for w in [.25,.5,1.]:
            pred=base.copy();pred[changed]=corrected(base[changed],delta[changed],w,conditional)
            score=roc_auc_score(y,pred.ravel());results.append({'weight':w,'conditional':conditional,'auc':float(score),'gain':float(score-auc)})
    a.out.mkdir(parents=True,exist_ok=True);np.savez_compressed(a.out/'correction.npz',selection=select,delta=delta,changed=changed)
    report={'baseline_auc':float(auc),'changed_events':int(changed.sum()),'results':results}
    (a.out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
