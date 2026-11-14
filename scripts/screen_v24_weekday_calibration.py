"""Train on held-out Thursday factor scores; evaluate untouched Friday once."""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from common import load_data, subset, binary_labels, metrics
from v24_causal_propensity import snapshot, first_events
from v24_weekday_calibration import build, train, predict, PARAMS, ROUNDS

THURSDAY = 1737590400
FRIDAY = 1737676800
DAY = 86400


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def checked_cohort(folder, data, cut):
    plan = json.loads((folder/'plan.json').read_text())
    if plan['cut'] != cut or plan['variant'] != 'joint' or plan['posterior_samples'] != 256:
        raise ValueError('Unexpected frozen-factor provenance.')
    if plan['core_recipe']['preference_epochs'] != 16 or plan['core_recipe']['residual_epochs'] != 17:
        raise ValueError('Factor core schedule changed.')
    if plan['joint_recipe']['epochs'] != 24 or plan['joint_recipe']['lr'] != .001:
        raise ValueError('Factor joint schedule changed.')
    file = folder/'predictions.npz'; z = dict(np.load(file))
    selected = (z['timestamp'] >= cut) & (z['timestamp'] < cut + DAY)
    ix = z['indices'][selected]
    expected = first_events(data, cut, cut+DAY)
    if len(np.unique(ix)) != len(ix) or not np.array_equal(np.sort(ix), np.sort(expected)):
        raise ValueError('Frozen factor predictions do not cover the required first-user/day cohort.')
    if not np.array_equal(z['labels'][selected], binary_labels(subset(data, ix))):
        raise ValueError('Frozen factor outcome alignment failed.')
    if not np.array_equal(z['user'][selected], data['user'][ix]):
        raise ValueError('Frozen factor user alignment failed.')
    p = z['joint'][selected]
    return ix, p, {'plan_sha256':digest(folder/'plan.json'), 'predictions_sha256':digest(file),
                   'checkpoint_sha256':digest(folder/'joint/residual.pt'), 'cut':cut,
                   'events':len(ix), 'min_query_timestamp':int(data['timestamp'][ix].min()),
                   'max_query_timestamp':int(data['timestamp'][ix].max())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ['root', 'thursday-factor', 'friday-factor', 'out']:
        p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--plan-only', action='store_true')
    a=p.parse_args(); a.out.mkdir(parents=True, exist_ok=True)
    source_root=Path(__file__).resolve().parents[1]
    plan={'design_revision':2, 'amendment':'Before any calibration fitting or Friday score access: precise later-transfer recipe and provenance frozen.',
          'training_cut':THURSDAY, 'training_label_window':[THURSDAY,FRIDAY],
          'validation_cut':FRIDAY, 'validation_label_window':[FRIDAY,FRIDAY+DAY],
          'factor_variant':'joint', 'factor_posterior_samples':256,
          'params':PARAMS, 'rounds':ROUNDS,
          'offset':'logit(frozen joint-factor click probability summed over ten candidates)',
          'features':'41 causal raw propensity features plus factor conditional entropy, max probability, event mass, and squared mass sum.',
          'selection':'One fixed fit; no early stopping, model/weight/seed selection, or Friday tuning.',
          'prediction':'Replace only factor event mass; preserve its item-choice probabilities.',
          'scope':'Standalone fresh weekday test, not an exact V20 weekday reproduction.',
          'future_step':'Weekend transfer requires review. Fit the same calibrator on Thursday+Friday out-of-time factors only, with no Saturday/Sunday labels; then fixed V20 event-mass mixture weights 0.125,0.25,0.5. No evaluation inference or submission in this script.',
          'factor_plan_sha256':{'thursday':digest(a.thursday_factor/'plan.json'),'friday':digest(a.friday_factor/'plan.json')},
          'data_sha256':digest(a.root/'data/train.npz'),
          'source_sha256':{s:digest(source_root/s) for s in ['src/v24_causal_propensity.py','src/v24_weekday_calibration.py','scripts/screen_v24_weekday_calibration.py']}}
    plan_file=a.out/'plan.json'
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise ValueError('Immutable experiment plan differs.')
    plan_file.write_text(json.dumps(plan,indent=2)+'\n')
    if a.plan_only:
        print(json.dumps(plan,indent=2)); return
    if (a.out/'report.json').exists() or (a.out/'model.txt').exists():
        raise ValueError('Existing fitted output: use the existing results; do not silently rerun.')
    started=time.time();data,_,audit=load_data(a.root/'data')
    ti,tp,tmeta=checked_cohort(a.thursday_factor,data,THURSDAY)
    state=snapshot(subset(data,np.flatnonzero(data['timestamp']<THURSDAY)),audit['n_items'],audit['n_users'],THURSDAY)
    query=subset(data,ti);query['target']=np.full(len(ti),-1)
    tx,to,names=build(query,state,tp);ty=(data['target'][ti]<10)
    assert data['timestamp'][ti].max()<FRIDAY
    model=train(tx,to,ty,names);model.save_model(str(a.out/'model.txt'))
    # Friday factors and labels are accessed only after the single training fit.
    vi,vp,vmeta=checked_cohort(a.friday_factor,data,FRIDAY)
    state=snapshot(subset(data,np.flatnonzero(data['timestamp']<FRIDAY)),audit['n_items'],audit['n_users'],FRIDAY)
    query=subset(data,vi);query['target']=np.full(len(vi),-1)
    vx,vo,vnames=build(query,state,vp);assert names==vnames
    candidate=predict(model,vx,vo,vp); labels=binary_labels(subset(data,vi));event_y=labels.sum(1)>0
    baseline_metrics=metrics(labels,vp);candidate_metrics=metrics(labels,candidate)
    p0=vp.sum(1);p1=candidate.sum(1)
    event_metrics={'baseline_auc':float(roc_auc_score(event_y,p0)),
                   'candidate_auc':float(roc_auc_score(event_y,p1)),
                   'baseline_logloss':float(log_loss(event_y,p0)),
                   'candidate_logloss':float(log_loss(event_y,p1))}
    checks={'columns':len(names),'training_rows':len(ti),'validation_rows':len(vi),
            'training_label_max_before_validation_cut':bool(data['timestamp'][ti].max()<FRIDAY),
            'no_overlap_outcome_indices':not bool(np.intersect1d(ti,vi).size),
            'within_slate_choice_max_abs_error':float(np.max(np.abs(candidate/p1[:,None]-vp/p0[:,None]))),
            'probabilities_finite':bool(np.isfinite(candidate).all()),
            'max_event_mass':float(candidate.sum(1).max()),'evaluation_targets_used':False}
    np.savez_compressed(a.out/'training_alignment.npz',indices=ti,event_labels=ty,factor_probability=tp)
    np.savez_compressed(a.out/'friday_predictions.npz',indices=vi,labels=labels.ravel(),baseline=vp.ravel(),candidate=candidate.ravel(),user=np.repeat(data['user'][vi],10),timestamp=data['timestamp'][vi])
    (a.out/'feature_schema.json').write_text(json.dumps(names,indent=2)+'\n')
    source_root=Path(__file__).resolve().parents[1]
    source_hashes={s:digest(source_root/s) for s in ['src/v24_causal_propensity.py','src/v24_weekday_calibration.py','scripts/screen_v24_weekday_calibration.py']}
    report={'plan':plan,'training':tmeta,'validation':vmeta,'baseline':baseline_metrics,'candidate':candidate_metrics,
            'global_auc_gain':candidate_metrics['auc']-baseline_metrics['auc'],
            'global_logloss_change':candidate_metrics['logloss']-baseline_metrics['logloss'],
            'event_metrics':event_metrics,'checks':checks,'source_sha256':source_hashes,
            'model_sha256':digest(a.out/'model.txt'),'data_sha256':digest(a.root/'data/train.npz'),
            'seconds':time.time()-started,'status':'review_required',
            'limitations':['Single fresh weekday holdout, not V20.','Thursday and Friday share users but only prior outcomes enter each factor or feature snapshot.','No hyperparameters selected using Friday outcomes.','No claim of public AUC gain.']}
    (a.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
