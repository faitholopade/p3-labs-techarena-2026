"""Build outer-fold-specific ensemble offsets, without held-out-user labels."""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch,lightgbm as lgb
from flexible_tree import predict as tree_predict
from calibration_model import predict as neural_predict
from infer_v13 import verify_hash


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['features','v20-extension','v20-alignment','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);raw=np.load(a.features/'raw.npy',mmap_mode='r');names=json.loads((a.features/'names.json').read_text())
    al=dict(np.load(a.features/'alignment.npz'));old=dict(np.load(a.v20_alignment));old_folds={int(u):int(f) for u,f in zip(old['user'],old['fold'])}
    assert all(old_folds[int(u)]==int(f) for u,f in zip(al['user'],al['fold']))
    model_spec=json.loads((a.v20_extension/'models/v20_spec.json').read_text());heads={h['name']:h for h in model_spec['heads']}
    tree_spec=heads['expanded700']['training_spec'];assert tree_spec['names']==names
    baseline=dict(np.load(a.features/'baseline_heads.npz'));records=[];started=time.time()
    for fold in [0,1,2,'full']:
        target=a.out/(f'fold_{fold}.npz' if fold!='full' else 'full.npz');metadata=target.with_suffix('.json')
        values={}
        for kind,name in [('tree','expanded700'),('neural','neural32')]:
            h=heads[name]
            if fold=='full':path=a.v20_extension/'models'/h['file'];expected=h['sha256']
            else:
                entry=h['validation_fold_models'][fold];path=a.v20_extension/entry['file'];expected=entry['sha256']
            verify_hash(path,expected)
            if kind=='tree':model=lgb.Booster(model_file=str(path));values[kind]=tree_predict(model,raw,tree_spec)
            else:model=torch.load(path,map_location='cpu',weights_only=False);values[kind]=neural_predict(model,raw)
            del model;gc.collect()
        parity={}
        if fold!='full':
            rows=np.repeat(al['fold']==fold,10)
            parity={key:float(np.max(abs(values[key][rows]-baseline[key][rows]))) for key in values};assert max(parity.values())<1e-6
        np.savez_compressed(target,**values)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()
        rec={'fold':fold,'file':target.name,'sha256':digest,'rows':len(raw),'heldout_prediction_parity':parity,'seconds':time.time()-started}
        metadata.write_text(json.dumps(rec,indent=2));records.append(rec);print(rec,flush=True)
    spec={'baseline':'v20_calibration_subensemble','formula':'(7*tree + 8*neural)/15','records':records,
          'user_fold_assignments_match_v20':True,'outer_validation_users_excluded_from_both_stages':True,
          'training_offset_note':'For each outer fold, the same frozen V20 heads predict both training and validation rows. Training offsets are in-sample for those frozen heads, as in residual boosting; validation users are excluded from both head and residual fitting.',
          'feature_spec':json.loads((a.features/'spec.json').read_text())}
    (a.out/'spec.json').write_text(json.dumps(spec,indent=2))


if __name__=='__main__':main()
