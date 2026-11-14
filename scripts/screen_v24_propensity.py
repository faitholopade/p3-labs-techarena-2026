"""Fixed causal propensity experiment, with fresh weekday and V20 comparisons."""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import lightgbm as lgb
from scipy.special import logit
from sklearn.metrics import roc_auc_score, log_loss
from common import load_data, subset
from v24_causal_propensity import first_events, snapshot, features, predict

START = 1737331200
DAY = 86400
CUTS = [1737676800, 1737763200, 1737849600, 1737936000]
WEIGHTS = [.125, .25, .5]
PARAMS = dict(objective='binary', metric='None', learning_rate=.025,
              num_leaves=7, min_data_in_leaf=1000, lambda_l2=100.,
              max_bin=127, feature_fraction=1., bagging_fraction=1.,
              num_threads=2, seed=20261029, verbosity=-1,
              force_col_wise=True, deterministic=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--incumbent', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    plan = {'cuts': CUTS, 'weights': WEIGHTS, 'params': PARAMS, 'rounds': 300,
            'training_cohorts': 'First event per user in each prior UTC day from Jan21; counters frozen at midnight.',
            'features': 'Causal raw user activity, hierarchical hour/day/context rates, provided history, slate-item rates; no fitted factor or user ID.',
            'evaluation_targets_used': False, 'no_early_stopping': True,
            'chronological_weekday_holdout': 1737676800,
            'limits': 'Fresh weekday model tests event propensity only; it is not an exact weekday V20 reproduction. Weekend comparison reuses development outcomes.'}
    plan_file = a.out / 'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('Plan changed: use a new output directory.')
    plan_file.write_text(json.dumps(plan, indent=2) + '\n')
    data, evaluation, audit = load_data(a.root / 'data')
    ev = dict(np.load(a.incumbent))
    evi = ev['indices']; period = ev['period'].reshape(-1, 10)[:, 0]
    assert set(np.unique(period)) == {0, 1}
    assert np.array_equal(np.repeat(period, 10), ev['period'])
    assert np.array_equal((np.arange(10)[None, :] == data['target'][evi, None]).ravel(), ev['labels'])
    started = time.time(); train_blocks = []; target_blocks = []; offsets = []
    row_days = []; train_indices = []; day_records = []
    for day in range(START + DAY, CUTS[-1], DAY):
        fit = subset(data, np.flatnonzero(data['timestamp'] < day))
        state = snapshot(fit, audit['n_items'], audit['n_users'], day)
        ix = first_events(data, day, day + DAY)
        query = subset(data, ix); query['target'] = np.full(len(ix), -1)
        x, baseline, names = features(query, state)
        train_blocks.append(x); target_blocks.append(data['target'][ix] < 10)
        offsets.append(baseline); row_days.append(np.full(len(ix), day)); train_indices.append(ix)
        day_records.append({'cut': day, 'events': len(ix), 'max_fit_timestamp': state['max_timestamp']})
        print({'features_ready': day, 'events': len(ix), 'seconds': time.time()-started}, flush=True)
    xall = np.concatenate(train_blocks); yall = np.concatenate(target_blocks)
    ball = np.concatenate(offsets); days = np.concatenate(row_days)
    np.savez_compressed(a.out/'training_alignment.npz', indices=np.concatenate(train_indices), days=days, labels=yall, baseline=ball)
    (a.out/'feature_schema.json').write_text(json.dumps(names, indent=2)+'\n')
    records = []; matched_event = np.full(len(evi), np.nan, dtype=np.float64)
    coverage = np.zeros(len(evi), dtype=np.int8)
    for cut in CUTS:
        folder = a.out / str(cut); folder.mkdir(exist_ok=True)
        training = days < cut
        model = lgb.train(PARAMS, lgb.Dataset(xall[training], label=yall[training],
                          init_score=logit(np.clip(ball[training], 1e-6, 1-1e-6)),
                          feature_name=names), num_boost_round=300)
        model.save_model(str(folder/'model.txt'))
        fit = subset(data, np.flatnonzero(data['timestamp'] < cut))
        state = snapshot(fit, audit['n_items'], audit['n_users'], cut)
        rec = {'cut': cut, 'training_events': int(training.sum()), 'max_training_label_timestamp': int(data['timestamp'][np.concatenate(train_indices)[training]].max()),
               'max_snapshot_timestamp': state['max_timestamp']}
        if cut != CUTS[-1]:
            ix = first_events(data, cut, cut+DAY); query = subset(data, ix); query['target']=np.full(len(ix),-1)
            x, base, new_names = features(query,state); assert names == new_names
            pred=predict(model,x,base); labels=data['target'][ix]<10
            rec.update(events=len(ix), baseline_auc=float(roc_auc_score(labels,base)), candidate_auc=float(roc_auc_score(labels,pred)),
                       baseline_logloss=float(log_loss(labels,base)),candidate_logloss=float(log_loss(labels,pred)))
            np.savez_compressed(folder/'first_day_predictions.npz',indices=ix,labels=labels,baseline=base,candidate=pred,user=data['user'][ix])
        else:
            query={k:v.copy() for k,v in evaluation.items()}; query['target']=np.full(len(query['user']),-1)
            x,base,new_names=features(query,state);assert names==new_names
            pred=predict(model,x,base)
            np.savez_compressed(folder/'evaluation_event_predictions.npz',event_probability=pred,baseline=base)
            rec['evaluation_events']=len(pred)
        if cut in CUTS[1:3]:
            pe = 0 if cut==CUTS[1] else 1; sel=np.flatnonzero(period==pe);ix=evi[sel]
            query=subset(data,ix);query['target']=np.full(len(ix),-1)
            x,base,new_names=features(query,state);assert names==new_names
            pred=predict(model,x,base);matched_event[sel]=pred;coverage[sel]+=1
            np.savez_compressed(folder/'matched_predictions.npz',indices=ix,event_probability=pred,baseline=base)
        rec['seconds']=time.time()-started;records.append(rec)
        (folder/'spec.json').write_text(json.dumps(rec,indent=2)+'\n');print(rec,flush=True)
    assert np.all(coverage == 1) and np.isfinite(matched_event).all()
    base=ev['candidate'].reshape(-1,10)
    assert np.isfinite(base).all() and np.all(base >= 0) and np.all(base.sum(1) > 0)
    replacement=base/base.sum(1,keepdims=True)*matched_event[:,None]
    assert np.allclose(replacement.sum(1), matched_event, rtol=1e-12, atol=1e-12)
    rows=[]
    for weight in WEIGHTS:
        candidate=((1-weight)*base+weight*replacement).ravel();gains={}
        for label,m in [('combined',np.ones(len(candidate),bool)),('earlier',ev['period']==0),('later',ev['period']==1)]:
            gains[label]=float(roc_auc_score(ev['labels'][m],candidate[m])-roc_auc_score(ev['labels'][m],ev['candidate'][m]))
        rows.append({'weight':weight,'gain_vs_v20':gains})
    np.savez_compressed(a.out/'matched_predictions.npz',indices=evi,event_probability=matched_event,replacement=replacement.ravel(),baseline=ev['candidate'],labels=ev['labels'],period=ev['period'],user=np.repeat(data['user'][evi],10))
    report={'plan':plan,'day_features':day_records,'models':records,'matched_results':rows,'seconds':time.time()-started,
            'metric_notes': {'models_auc': 'Event click-vs-no-click ROC AUC; baseline is the causal smoothed user click rate, not V20.',
                             'matched_results': 'Global candidate-row ROC AUC gains over V20, preserving its conditional item choice.'},
            'decision': 'reject' if all(r['gain_vs_v20']['combined'] <= 0 for r in rows) else 'review',
            'no_submission_generated': True}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
