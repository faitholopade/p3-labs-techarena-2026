"""Validate V21 submission serialization, baseline identity, and model provenance."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd,torch
from validate_submission import validate


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['csv','rowids','components','extension','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);ids=np.load(a.rowids);ev=dict(np.load(a.components));spec=json.loads((a.extension/'models/v21_spec.json').read_text())
    assert np.array_equal(ev['rowid'].astype(str),ids.astype(str));members=[h['name'] for h in spec['heads']]
    prediction=ev['v20']+spec['replacement_weight']*(np.mean([ev[k] for k in members],axis=0)-ev['v20_neural'])
    assert np.array_equal(prediction,ev['prediction'])
    def csv_sha(values):return hashlib.sha256(pd.DataFrame({'rowid':ids,'predicted_probability':values}).to_csv(index=False,float_format='%.10g').encode()).hexdigest()
    sha=csv_sha(prediction);assert sha==digest(a.csv);parent_sha=csv_sha(ev['v20']);assert parent_sha==spec['required_v20_csv_sha256']
    repo=Path(__file__).resolve().parents[1]
    for name,expected in spec['source_file_sha256'].items():assert digest(repo/name)==expected,name
    for h in spec['heads']:
        model=a.extension/'models'/h['file'];assert digest(model)==h['sha256']
        m=torch.load(model,map_location='cpu',weights_only=False);assert m['train_events']==spec['feature_spec']['events']
        for f in h['validation_fold_models']:assert digest(a.extension/f['file'])==f['sha256']
    report=validate(a.csv,ids);rounded=pd.read_csv(a.csv).predicted_probability.to_numpy();report.update(version='v21',events=len(rounded)//10,max_event_probability=float(rounded.reshape(-1,10).sum(1).max()))
    assert report['rows']==529390 and report['reference_order_matches'] and report['max_event_probability']<=1+1e-8
    reproduction={'baseline':'v20','baseline_sha256':parent_sha,'scored_baseline_reproduced':True,'component_recomposition_sha256':sha,'component_recomposition_identical':True,'full_baseline_inference_runs':1,'limitation':'Independent recomposition of saved component predictions, not a second full inference run.'}
    for name,obj in [('submission_validation',report),('inference_reproduction',reproduction),('source_verification',{'source_files_verified':len(spec['source_file_sha256']),'heads_verified':len(spec['heads']),'fold_models_verified':3*len(spec['heads'])})]:
        (a.out/f'v21_{name}.json').write_text(json.dumps(obj,indent=2)+'\n')
    print(json.dumps({'validation':report,'reproduction':reproduction},indent=2),flush=True)


if __name__=='__main__':main()
