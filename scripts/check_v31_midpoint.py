"""One fixed V27/V29 midpoint, with the established weekday and forward controls."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
from sklearn.metrics import roc_auc_score,log_loss
from backtest_v24_factor import sha256,write_json
from screen_v30_pooled import atomic_npz

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v27','v29','weekday-audit','weekday-features','forward','plan','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text());base=dict(np.load(a.v27));new=dict(np.load(a.v29))
    for key in ['indices','labels','period','users']:assert np.array_equal(base[key],new[key])
    assert np.array_equal(base['candidate'],new['baseline']);candidate=.5*base['candidate']+.5*new['candidate']
    masks={'combined':np.ones(len(candidate),bool),'earlier':base['period']==0,'later':base['period']==1}
    gains={key:float(roc_auc_score(base['labels'][m],candidate[m])-roc_auc_score(base['labels'][m],base['candidate'][m])) for key,m in masks.items()}
    atomic_npz(a.out/'evidence.npz',indices=base['indices'],labels=base['labels'],baseline=base['candidate'],candidate=candidate,period=base['period'],users=base['users'])
    audit=json.loads(a.weekday_audit.read_text());weekday={r['cohort']:r['half_v29_half_v27_head_gain'] for r in audit['v29_weekday_diagnostic']}
    forward=np.load(a.forward/'all_predictions.npz');control=forward['base_20261029'];memory=forward['memory_20261029'];half=.5*control+.5*memory
    al=np.load(a.weekday_features/'friday/alignment.npz');pos=al['first_position'];rows=(pos[:,None]*10+np.arange(10)).ravel();labels=al['labels'][rows]
    for name in ['base_20261029','memory_20261029']:
        completed=json.loads((a.forward/(name+'_completed.json')).read_text());assert completed['model_sha256']==sha256(a.forward/(name+'.pt'))
    forward_gain=float(roc_auc_score(labels,half)-roc_auc_score(labels,control))
    atomic_npz(a.out/'forward_evidence.npz',indices=al['indices'][pos],labels=labels,baseline=control,candidate=half,users=np.repeat(al['user'][pos],10))
    loss=float(log_loss(base['labels'],candidate)-log_loss(base['labels'],base['candidate']))
    passes=all(gains[k]>0 for k in ['earlier','later']) and gains['combined']>=plan['minimum_exploratory_gain'] and loss<=0 and all(v>0 for v in weekday.values()) and forward_gain>=0
    report={'plan':plan,'source_sha256':sha256(Path(__file__)),'inputs_sha256':{'v27':sha256(a.v27),'v29':sha256(a.v29),'weekday_audit':sha256(a.weekday_audit),'forward_predictions':sha256(a.forward/'all_predictions.npz')},
            'matched_gains':gains,'matched_logloss_change':loss,'weekday_half_head_gains':weekday,'forward_half_head_gain':forward_gain,
            'forward_logloss_change':float(log_loss(labels,half)-log_loss(labels,control)),'passes_before_bootstrap':passes,'portal_uploads':0}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':main()
