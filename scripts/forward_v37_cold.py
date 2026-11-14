"""Fixed V37 Friday component check using Thursday-only calibration heads."""
from screen_v37_cold import *

def forward(a, plan):
    selected=json.loads((a.out/'report.json').read_text())['selected']
    quality=json.loads((a.out.parent/'assessment/decision.json').read_text())
    assert selected and quality['incumbent']=='v33' and quality['positive_bootstrap_lower_bound']
    w=a.workspace; d=dict(np.load(a.out/'friday/predictions.npz'))
    f=np.load(a.out/'friday/changed_features.npz'); raw=f['raw']; changed=f['changed']; rows=(f['at'][:,None]*10+np.arange(10)).ravel()
    folder=w/'research_v25/weekday_stress'; saved=np.load(folder/'all_predictions.npz')
    small_folder=w/'research_v33/screen/small_memory/forward'
    small=np.load(small_folder/'predictions.npz'); assert np.array_equal(small['indices'],d['indices']) and np.array_equal(small['labels'],d['labels'])
    old=[saved['base_20261029'],saved['memory_20261029'],small['new_head']]
    paths=[folder/'base_20261029.pt',folder/'memory_20261029.pt',small_folder/'model.pt']
    reports=[folder/'base_20261029_completed.json',folder/'memory_20261029_completed.json',small_folder/'report.json']
    new=[]; checks=[]
    for j,(path,report) in enumerate(zip(paths,reports)):
        info=json.loads(report.read_text()); assert sha256(path)==info['model_sha256']
        h=torch.load(path,map_location='cpu',weights_only=False); cols=344 if j==0 else 388
        assert h['names']==f['names'][:cols].tolist()
        error=float(np.max(abs(predict(h,raw[:,:cols])-old[j][rows])))
        assert error<1e-7
        p=old[j].copy();p[rows]=predict(h,changed[:,:cols]);new.append(p)
        checks.append({'model':str(path),'sha256':sha256(path),'old_prediction_error':error})
    baseline=sum(w*p for w,p in zip(WEIGHTS,old))/sum(WEIGHTS)
    candidate=baseline+selected['alpha']*sum(w*(n-o) for w,n,o in zip(WEIGHTS,new,old))/sum(WEIGHTS)
    assert np.array_equal(candidate[~d['changed']],baseline[~d['changed']])
    report={'alpha':selected['alpha'],'baseline_auc':float(roc_auc_score(d['labels'],baseline)),
            'candidate_auc':float(roc_auc_score(d['labels'],candidate)),
            'gain':float(roc_auc_score(d['labels'],candidate)-roc_auc_score(d['labels'],baseline)),
            'logloss_change':float(log_loss(d['labels'],candidate)-log_loss(d['labels'],baseline)),
            'head_checks':checks,'established_users_unchanged':True,
            'scope':'First-Friday changed pooled heads; calibrators trained on Thursday only, factor and adaptation chronological before Friday. Not full V33.'}
    out=a.out/'forward';out.mkdir(exist_ok=True)
    atomic_npz(out/'predictions.npz',indices=d['indices'],labels=d['labels'],users=d['users'],baseline=baseline,candidate=candidate)
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['workspace','plan','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();forward(a,json.loads(a.plan.read_text()))

if __name__=='__main__':main()
