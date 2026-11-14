"""Fixed full refits and Thursday-only diagnostics for the authorized V38 attempt."""
import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
from daily_training import train
from calibration_model import predict as neural_predict
import flexible_tree
import v30_tree
from screen_v29_pooled import load_datasets
from screen_v30_pooled import position, atomic_npz
from check_v36_combination import changed_component
from backtest_v24_factor import sha256, write_json


def predict_events(model, raw, events, kind, spec=None):
    parts = []
    for start in range(0, len(events), 2000):
        rows = (events[start:start+2000, None]*10+np.arange(10)).ravel()
        block = raw[rows]
        if kind == 'slate':
            value = neural_predict(model, block)
        elif kind == 'tree':
            value = v30_tree.predict(model, block, spec)
        else:
            value = flexible_tree.predict(model, block, spec)
        parts.append(value)
    return np.concatenate(parts)


def fit(a, plan):
    kind, stage = a.stage.split('_')
    full = stage == 'full'
    if kind == 'legacy' and full:
        raise ValueError('There is no new legacy full fit.')
    out = a.out / a.stage; out.mkdir(parents=True, exist_ok=True)
    ds, align = load_datasets(a, 388 if kind == 'slate' else 344)
    config = plan[{'legacy':'legacy_tree_config', 'tree':'tree_config', 'slate':'slate_config'}[kind]]
    indices = [0,1,2] if full else [0]
    inputs = {'stage':a.stage,'plan_sha256':sha256(a.plan),'config':config,
              'selection_sha256':sha256(a.research/'screen/report.json'),
              'assessment_sha256':sha256(a.research/'assessment/decision.json'),
              'feature_report_sha256':[sha256(f/'report.json') for f in [a.weekdays/'thursday',a.weekdays/'friday',a.weekends]],
              'alignment_sha256':[sha256(f/'alignment.npz') for f in [a.weekdays/'thursday',a.weekdays/'friday',a.weekends]],
              'source_sha256':{str(p.relative_to(REPO)):sha256(p) for p in [Path(__file__),REPO/'src/daily_training.py',REPO/'src/calibration_model.py',REPO/'src/flexible_tree.py',REPO/'src/v30_tree.py']}}
    if (out/'plan.json').exists() and json.loads((out/'plan.json').read_text()) != inputs:
        raise ValueError('Frozen refit changed.')
    write_json(out/'plan.json', inputs)
    if (out/'report.json').exists():
        report = json.loads((out/'report.json').read_text())
        assert sha256(out/report['model_file']) == report['model_sha256']
        print(json.dumps(report),flush=True); return
    modelpath = out / ('model.pt' if kind == 'slate' else 'model.txt')
    if modelpath.exists():
        raise ValueError('Unverified model already exists.')
    if kind == 'tree':
        needed = sum(len(align[k]['user']) for k in indices)*10*269*4
        if shutil.disk_usage('/tmp').free < needed + 150_000_000:
            raise ValueError('Insufficient temporary disk for fixed tree fit.')
    started=time.time(); print({'stage':a.stage,'training_events':sum(len(align[k]['user']) for k in indices)},flush=True)
    masks = [np.ones(len(align[k]['user']),bool) for k in indices]
    if kind == 'slate':
        model=train([ds[k] for k in indices],masks,**config); spec=None
        temporary=out/'model.partial';torch.save(model,temporary);temporary.replace(modelpath)
    elif kind == 'tree':
        model,spec=v30_tree.fit([ds[k] for k in indices],masks,config)
        temporary=out/'model.partial';model.save_model(str(temporary));temporary.replace(modelpath)
    else:
        model,spec=flexible_tree.fit(ds[0]['x'],ds[0]['labels'],ds[0]['names'].tolist(),config)
        temporary=out/'model.partial';model.save_model(str(temporary));temporary.replace(modelpath)
    if spec is not None:write_json(out/'training_spec.json',spec)
    report={'inputs':inputs,'model_file':modelpath.name,'model_sha256':sha256(modelpath),
            'training_events':sum(int(m.sum()) for m in masks),'seconds':time.time()-started}
    if not full:
        pos=position(align[1]); rows=(pos[:,None]*10+np.arange(10)).ravel()
        prediction=predict_events(model,ds[1]['x'],pos,kind,spec)
        atomic_npz(out/'predictions.npz',prediction=prediction,indices=align[1]['indices'][pos],
                   labels=align[1]['labels'][rows],users=np.repeat(align[1]['user'][pos],10))
        report['prediction_sha256']=sha256(out/'predictions.npz')
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


def forward_check(a, plan):
    records={}; predictions={}
    for kind in ['legacy','tree','slate']:
        folder=a.out/(kind+'_forward');report=json.loads((folder/'report.json').read_text())
        assert sha256(folder/report['model_file']) == report['model_sha256']
        assert sha256(folder/'predictions.npz') == report['prediction_sha256']
        assert report['training_events']==97717
        records[kind]=report; predictions[kind]=dict(np.load(folder/'predictions.npz'))
    ev=predictions['legacy']
    for other in ['tree','slate']:
        for key in ['indices','labels','users']:assert np.array_equal(ev[key],predictions[other][key])
    controls=np.load(a.forward_controls/'all_predictions.npz')
    old=[];control_hashes={}
    for name in ['base_20261029','memory_20261029']:
        report=json.loads((a.forward_controls/(name+'_completed.json')).read_text())
        assert report['model_sha256']==sha256(a.forward_controls/(name+'.pt'))
        old.append(controls[name]);control_hashes[name]=report['model_sha256']
    report=json.loads((a.small_forward/'report.json').read_text())
    assert report['model_sha256']==sha256(a.small_forward/'model.pt')
    small=dict(np.load(a.small_forward/'predictions.npz'))
    for key in ['indices','labels','users']:assert np.array_equal(ev[key],small[key])
    h33=(.109375*old[0]+.109375*old[1]+.125*small['new_head'])/.34375
    baseline,candidate=changed_component(ev['prediction'],predictions['tree']['prediction'],h33,predictions['slate']['prediction'],'tree_plus_slate')
    y=ev['labels'];report={'recipe':'tree_plus_slate','baseline_auc':float(roc_auc_score(y,baseline)),
           'candidate_auc':float(roc_auc_score(y,candidate)),
           'gain':float(roc_auc_score(y,candidate)-roc_auc_score(y,baseline)),
           'logloss_change':float(log_loss(y,candidate)-log_loss(y,baseline)),
           'fit_reports_sha256':{k:sha256(a.out/(k+'_forward')/'report.json') for k in records},
           'existing_control_hashes':control_hashes,'small_control_sha256':sha256(a.small_forward/'model.pt'),
           'scope':plan['forward_check'],'not_full_incumbent_validation':True,
           'release_mode':plan['release_mode'],'portal_uploads':0}
    atomic_npz(a.out/'forward_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],baseline=baseline,candidate=candidate)
    write_json(a.out/'forward_report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['weekdays','weekends','research','plan','out','forward-controls','small-forward']:
        p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--stage',required=True,choices=['legacy_forward','tree_forward','slate_forward','forward_check','tree_full','slate_full'])
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text())
    selection=json.loads((a.research/'screen/report.json').read_text())['selected']
    assert selection and selection['recipe']=='tree_plus_slate'
    assert plan['selected_research']=='V36 tree_plus_slate, unchanged'
    if a.stage=='forward_check':forward_check(a,plan)
    else:fit(a,plan)


if __name__=='__main__':main()
