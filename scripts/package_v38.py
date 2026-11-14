"""Verify V38 twice and package exactly one CSV for manual competition upload."""
import argparse
import hashlib
import json
import os
import sys
import zipfile
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'src'))
import numpy as np
import pandas as pd
from infer_v38 import sha,serialize,convex_rebuild


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['csv','recheck','reference','spec','report']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();digest=sha(a.csv);assert digest==sha(a.recheck)
    path=a.csv.with_suffix('.components.npz')
    with zipfile.ZipFile(path) as z:assert z.testzip() is None
    parts=dict(np.load(path));spec=json.loads(a.spec.read_text());assert spec['version']=='v38'
    assert digest!=spec['required_v33_csv_sha256'], 'Candidate is identical to scored V33'
    reference=np.load(a.reference);frame=pd.read_csv(a.csv)
    assert frame.columns.tolist()==['rowid','predicted_probability']
    assert len(frame)==529390 and frame.rowid.nunique()==529390
    assert np.array_equal(reference,frame.rowid.to_numpy()) and np.array_equal(reference,parts['rowid'])
    prob=frame.predicted_probability.to_numpy()
    assert np.isfinite(prob).all() and prob.min()>=0 and prob.max()<=1
    assert prob.reshape(-1,10).sum(1).max()<=1+1e-8
    exact=.875*(parts['v33']+.07177734375*(parts['v38_tree']-parts['expanded700']))+.125*parts['v38_slate']
    assert np.array_equal(exact,parts['candidate'])
    assert hashlib.sha256(serialize(reference,exact).encode()).hexdigest()==digest
    convex=convex_rebuild(parts);error=float(np.max(abs(exact-convex)));assert error<5e-16
    report=json.loads(a.csv.with_suffix('.validation.json').read_text())
    repeated=json.loads(a.recheck.with_suffix('.validation.json').read_text())
    for validation in [report,repeated]:
        assert validation['spec_sha256']==sha(a.spec) and validation['csv_sha256']==digest
        for parent in ['v20','v27','v29','v31','v33']:
            assert validation['scored_parent_hashes'][parent]==spec['required_'+parent+'_csv_sha256']
    destination=a.csv.with_suffix('.csv.zip');temporary=destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        info=zipfile.ZipInfo(a.csv.name,date_time=(2026,10,29,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED
        z.writestr(info,a.csv.read_bytes())
    with zipfile.ZipFile(temporary) as z:
        assert z.namelist()==[a.csv.name] and z.testzip() is None
        assert hashlib.sha256(z.read(a.csv.name)).hexdigest()==digest
    os.replace(temporary,destination)
    verification={'version':'v38','rows':len(frame),'filename':destination.name,'bytes':destination.stat().st_size,
      'sha256':sha(destination),'csv_sha256':digest,'files':1,'zip_crc_valid':True,'reference_order_matches':True,
      'finite_probabilities_in_range':True,'second_inference_exact':True,'component_formula_exact':True,
      'convex_max_abs_error':error,'scored_parents_reproduced':report['scored_parent_hashes'],
      'mean_absolute_probability_change_vs_v33':float(np.mean(abs(parts['candidate']-parts['v33']))),
      'max_absolute_probability_change_vs_v33':float(np.max(abs(parts['candidate']-parts['v33']))),
      'release_mode':spec['release_mode'],'normal_quality_gate':False,'positive_bootstrap_lower_bound':False,
      'limits':'Two exact reruns use the verified V20 cache; the historical parent chain was not retrained. Exploratory release with uncertainty interval crossing zero.',
      'purpose':'Competition upload: exactly one CSV.','official_score':None,'portal_uploads':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':main()
