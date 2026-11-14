"""Past-history adaptation for cold users in the released pooled calibration heads."""
import argparse
import gc
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'scripts')]
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, log_loss
from common import load_data, subset
from residual_model import load
from negative import pooled
from adapt_query_history import adapt
from factor_ensemble import update_features
from history_tree_features import extend
from calibration_model import predict
from screen_v29_pooled import load_datasets
from screen_v30_pooled import atomic_npz, position
from backtest_v24_factor import sha256, write_json

WEIGHTS = np.array([.109375, .109375, .125])
LABELS = ['base', 'memory', 'small']


def sources(w, cohort):
    if cohort in ['thursday', 'friday']:
        return (w / f'research_v27/{cohort}/users/residual.pt',
                w / f'research_v24/experiments/{cohort}_factor/bases/exposure.npz',
                w / f'research_v24/experiments/{cohort}_factor/pools')
    cut = {'earlier': 1737763200, 'later': 1737849600}[cohort]
    return (w / f'research_v24/runtime/P3_Labs_v17_models/validation/factors/{cohort}_factor_users_only.pt',
            w / f'research_v24/runtime/P3_Labs_v13_models/validation/source_states/{cut}/exposure.npz',
            w / 'research_v37/pools')


def controls(w):
    b = dict(np.load(w / 'research_v30/screen/controls/v27_predictions.npz'))
    m = dict(np.load(w / 'research_v30/screen/controls/v29_weekday_predictions.npz'))
    m['weekend'] = np.load(w / 'research_v29/screen/pooled_memory/predictions.npz')['prediction']
    s = dict(np.load(w / 'research_v33/screen/small_memory/predictions.npz'))
    return [b, m, s]


def head_folders(w):
    return [w / 'research_v27/pooled', w / 'research_v29/screen/pooled_memory',
            w / 'research_v33/screen/small_memory']


def fit_cohort(a, plan):
    w = a.workspace; cohort = a.cohort; k = plan['cohorts'].index(cohort); cut = plan['cutoffs'][k]
    out = a.out / cohort; out.mkdir(parents=True, exist_ok=True)
    ds, align = load_datasets(a, 388); di = k if k < 2 else 2
    pos = position(align[di]); selector = np.arange(len(pos))
    if k >= 2:
        selector = np.flatnonzero(align[di]['period'][pos] == k - 2)
        pos = pos[selector]
    indices = align[di]['indices'][pos]
    root = w / 'research_v24/runtime/P3_Labs_v11_model_and_code'
    data, _, audit = load_data(root / 'data')
    fit = subset(data, np.flatnonzero(data['timestamp'] < cut)); query = subset(data, indices)
    query['target'] = np.full(len(indices), -1)
    assert len(np.unique(query['user'])) == len(indices)
    assert np.all((query['hist'] == 0) | (query['hist_ts'] < cut))
    assert np.array_equal(query['user'], align[di]['user'][pos])
    factor, basis, pool_cache = sources(w, cohort)
    ck = torch.load(factor, map_location='cpu', weights_only=False); assert ck['cut'] == cut
    count = np.bincount(fit['user'], minlength=len(ck['state']['user.weight']))
    cold = (count[query['user']] == 0) & (query['hist'] > 0).any(1)
    at = np.flatnonzero(cold); cold_query = subset(query, at)
    assert len(at) > 0
    full_rows = (pos[:, None] * 10 + np.arange(10)).ravel()
    cold_rows = (at[:, None] * 10 + np.arange(10)).ravel()
    raw = ds[di]['x'][full_rows[cold_rows]].copy(); names = ds[di]['names'].tolist()
    assert len(names) == 388
    source_report = json.loads((a.weekdays / cohort / 'report.json').read_text()) if k < 2 else json.loads((a.weekends / 'report.json').read_text())['checks'][k - 2]
    assert sha256(factor) == source_report['factor_sha256']
    inputs = {'cohort': cohort, 'cut': cut, 'plan_sha256': sha256(a.plan), 'factor_sha256': sha256(factor),
              'exposure_sha256': sha256(basis), 'data_sha256': sha256(root / 'data/train.npz'),
              'alignment_sha256': sha256((a.weekdays / cohort if k < 2 else a.weekends) / 'alignment.npz'),
              'source_sha256': {str(p.relative_to(REPO)): sha256(p) for p in
                [Path(__file__), REPO/'src/adapt_query_history.py', REPO/'src/factor_ensemble.py',
                 REPO/'src/history_tree_features.py', REPO/'src/negative.py', REPO/'src/calibration_model.py']}}
    if (out / 'plan.json').exists():
        assert json.loads((out / 'plan.json').read_text()) == inputs
    write_json(out / 'plan.json', inputs)
    # Rebuilding the unchanged source first catches column/schema and posterior discrepancies.
    original = raw[:, :284].copy(); update_features(original, cold_query, factor, posterior_samples=256)
    original, original_names = extend(original, cold_query, ck['state'], names[:284])
    assert original_names == names[:344]
    parity = float(np.nanmax(np.abs(original - raw[:, :344])))
    assert np.allclose(original, raw[:, :344], rtol=0, atol=plan['feature_parity_tolerance'], equal_nan=True), parity
    pool = pooled(fit, audit['n_items'], audit['n_users'], cut, basis=basis, cache=pool_cache)
    inputs['pool_sha256'] = sha256(pool_cache / f'exposure_pools_{cut}_64.npz')
    started = time.time(); model_path = out / 'factor.pt'
    if (out / 'adaptation.json').exists():
        adaptation = json.loads((out / 'adaptation.json').read_text())
        assert sha256(model_path) == adaptation['model_sha256']
    else:
        assert not model_path.exists()
        net = load(factor)
        net, adaptation = adapt(net, fit, cold_query, pool, cut, mode='cluster', steps=120, seed=20261031, kl_weight=1.)
        state = net.state_dict(); unaffected = np.ones(len(state['user.weight']), bool); unaffected[query['user'][cold]] = False
        for key, value in ck['state'].items():
            if key in ['user.weight', 'user_logstd.weight']:
                assert torch.equal(state[key][unaffected], value[unaffected])
            else:
                assert torch.equal(state[key], value), key
        tmp = out / 'factor.partial'; torch.save(dict(ck, state=state), tmp); tmp.replace(model_path)
        adaptation.update(model_sha256=sha256(model_path), seconds=time.time()-started,
                          only_cold_preference_parameters_changed=True)
        write_json(out / 'adaptation.json', adaptation)
        del net
    newck = torch.load(model_path, map_location='cpu', weights_only=False)
    changed = raw[:, :284].copy(); update_features(changed, cold_query, model_path, posterior_samples=256)
    changed, changed_names = extend(changed, cold_query, newck['state'], names[:284])
    assert changed_names == names[:344]
    changed = np.c_[changed, raw[:, 344:]]
    ctl = controls(w); label = cohort if k < 2 else 'weekend'; select_rows = (selector[:, None]*10+np.arange(10)).ravel()
    result = {'indices': indices, 'labels': align[di]['labels'][full_rows],
              'users': np.repeat(query['user'], 10), 'changed': np.repeat(cold, 10)}
    heads = []; errors = {}
    for j, (name, folder) in enumerate(zip(LABELS, head_folders(w))):
        old = ctl[j][label][select_rows].copy(); new = old.copy(); cols = 344 if j == 0 else 388
        report = json.loads((folder/'report.json').read_text())
        for fold in range(3):
            events = np.flatnonzero(align[di]['fold'][pos[at]] == fold)
            rows = (events[:,None]*10+np.arange(10)).ravel()
            if not len(rows):
                continue
            checkpoint = folder/f'fold_{fold}.pt'
            expected = next(r['model_sha256'] for r in report['folds'] if r['fold']==fold)
            assert sha256(checkpoint) == expected
            h = torch.load(checkpoint,map_location='cpu',weights_only=False); assert h['names']==names[:cols]
            old_check = predict(h,raw[rows,:cols])
            error = float(np.max(abs(old_check-old[cold_rows[rows]]))) if len(rows) else 0.
            assert error < 1e-7
            new[cold_rows[rows]] = predict(h,changed[rows,:cols])
            errors[f'{name}_{fold}'] = error; heads.append({'path':str(checkpoint),'sha256':expected})
        assert np.array_equal(new[~result['changed']], old[~result['changed']])
        result[name+'_old'] = old; result[name+'_new'] = new
    atomic_npz(out/'predictions.npz',**result)
    # Save only the changed feature rows for a selected forward check; no full raw cache copy.
    atomic_npz(out/'changed_features.npz', raw=raw, changed=changed, at=at, names=np.asarray(names))
    write_json(out/'report.json',{'inputs':inputs,'events':len(indices),'cold_events':len(at),
               'adaptation':adaptation,'source_feature_max_error':parity,'head_parity_errors':errors,
               'head_checkpoints':heads,'established_users_unchanged':True,'query_targets_masked':True,
               'max_fit_timestamp':int(fit['timestamp'].max()),'seconds':time.time()-started})
    print(json.dumps({'cohort':cohort,'cold_events':len(at),'adaptation':adaptation,'feature_parity':parity}),flush=True)


def aggregate(a, plan):
    ev = dict(np.load(a.workspace/'research_v33/screen/selected_evidence.npz'))
    d = {label:dict(np.load(a.out/label/'predictions.npz')) for label in plan['cohorts']}
    weekend = {key:np.concatenate([d['earlier'][key],d['later'][key]]) for key in d['earlier']}
    for key in ['indices','labels','users']: assert np.array_equal(weekend[key],ev[key])
    def correction(data):
        return sum(w*(data[n+'_new']-data[n+'_old']) for w,n in zip(WEIGHTS,LABELS))
    def head(data): return sum(w*data[n+'_old'] for w,n in zip(WEIGHTS,LABELS))/sum(WEIGHTS)
    baseline=ev['candidate']; y=ev['labels']; delta=correction(weekend)
    masks={'combined':np.ones(len(y),bool),'earlier':ev['period']==0,'later':ev['period']==1}
    records=[]; candidates={}
    for alpha in plan['recipes']:
        candidate=baseline+alpha*delta
        assert np.isfinite(candidate).all() and candidate.min()>=0 and candidate.max()<=1
        assert candidate.reshape(-1,10).sum(1).max()<=1+1e-8
        assert np.array_equal(candidate[~weekend['changed']],baseline[~weekend['changed']])
        gains={name:float(roc_auc_score(y[mask],candidate[mask])-roc_auc_score(y[mask],baseline[mask])) for name,mask in masks.items()}
        weekdays={}
        for label in ['thursday','friday']:
            h=head(d[label]); new=h+alpha*correction(d[label])/sum(WEIGHTS)
            weekdays[label]=float(roc_auc_score(d[label]['labels'],new)-roc_auc_score(d[label]['labels'],h))
        loss=float(log_loss(y,candidate)-log_loss(y,baseline))
        eligible=gains['combined']>=1e-5 and gains['earlier']>0 and gains['later']>0 and loss<=0 and all(v>0 for v in weekdays.values())
        records.append({'alpha':alpha,'gains':gains,'weekday_head_gains':weekdays,'logloss_change':loss,'eligible_before_confirmation':eligible})
        candidates[alpha]=candidate
    eligible=[r for r in records if r['eligible_before_confirmation']]; selected=None
    if eligible:
        best=max(r['gains']['combined'] for r in eligible)
        selected=min([r for r in eligible if r['gains']['combined']>=best-5e-6],key=lambda r:r['alpha'])
        atomic_npz(a.out/'selected_evidence.npz',indices=ev['indices'],labels=y,users=ev['users'],period=ev['period'],baseline=baseline,candidate=candidates[selected['alpha']])
    report={'plan':plan,'results':records,'selected':selected,'portal_uploads':0,'official_score':None,
            'established_users_unchanged':True,'cohort_report_sha256':{k:sha256(a.out/k/'report.json') for k in d}}
    write_json(a.out/'report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','weekdays','weekends','plan','out']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cohort',choices=['thursday','friday','earlier','later']);p.add_argument('--aggregate',action='store_true')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);plan=json.loads(a.plan.read_text())
    if a.aggregate:aggregate(a,plan)
    else:fit_cohort(a,plan)


if __name__=='__main__':main()
