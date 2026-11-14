"""Independently recompose V20 saved predictions and validate its serialized CSV."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from validate_submission import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['csv','rowids','components','extension','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);ids=np.load(a.rowids);ev=np.load(a.components);spec=json.loads((a.extension/'models/v20_spec.json').read_text())
    assert np.array_equal(ev['rowid'].astype(str),ids.astype(str))
    prediction=sum(float(w)*ev[name] for name,w in zip(spec['weight_order'],spec['weights']) if w>0)
    assert np.array_equal(prediction,ev['prediction'])
    def csv_sha(values):
        data=pd.DataFrame({'rowid':ids,'predicted_probability':values}).to_csv(index=False,float_format='%.10g').encode()
        return hashlib.sha256(data).hexdigest()
    sha=csv_sha(prediction);assert sha==hashlib.sha256(a.csv.read_bytes()).hexdigest()
    parent_sha=csv_sha(ev['v19']);assert parent_sha==spec['required_v19_csv_sha256']
    report=validate(a.csv,ids);rounded=pd.read_csv(a.csv).predicted_probability.to_numpy()
    report.update({'version':'v20','events':len(rounded)//10,'columns':['rowid','predicted_probability'],'max_event_probability':float(rounded.reshape(-1,10).sum(1).max())})
    assert report['rows']==529390 and report['reference_order_matches'] and report['max_event_probability']<=1+1e-8
    reproduction={'version':'v20','full_inference_runs':1,'baseline':'v19','baseline_sha256':parent_sha,'scored_baseline_reproduced':True,
                  'component_recomposition_sha256':sha,'component_recomposition_identical':True,'max_component_recomposition_error':float(np.max(abs(prediction-ev['prediction']))),
                  'weights':spec['weights'],'weight_order':spec['weight_order'],'limitation':'Independent recomposition of saved predictions, not a second complete inference run.'}
    (a.out/'v20_submission_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    (a.out/'v20_inference_reproduction.json').write_text(json.dumps(reproduction,indent=2)+'\n')
    print(json.dumps({'validation':report,'reproduction':reproduction},indent=2),flush=True)


if __name__=='__main__':main()
