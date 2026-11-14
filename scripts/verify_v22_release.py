"""Check release provenance and independently recompose the saved predictions."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
from validate_submission import validate


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def csv_digest(ids,prediction):
    text=pd.DataFrame({'rowid':ids,'predicted_probability':prediction}).to_csv(index=False,float_format='%.10g')
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['extension','components','submission','rowids','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1];spec=json.loads((a.extension/'models/v22_spec.json').read_text());d=dict(np.load(a.components))
    for file,sha in spec['source_file_sha256'].items():
        if digest(repo/file)!=sha:raise ValueError('Source hash differs: '+file)
    count=0
    for head in spec['heads']:
        if digest(a.extension/'models'/head['file'])!=head['sha256']:raise ValueError('Full model hash differs.')
        count+=1
        for fold in head['validation_fold_models']:
            for part in fold.values():
                if digest(a.extension/part['file'])!=part['sha256']:raise ValueError('Fold artifact hash differs.')
                count+=1
    ids=d['rowid'];v20=(49/64)*d['v17']+(7/64)*d['v20_tree']+(1/8)*d['v20_neural']
    v21=d['v20']+(1/16)*(d['first_neural_ce']-d['v20_neural'])
    reconstructed=(7/8)*d['v21']+(1/8)*d['residual_binary3']
    if csv_digest(ids,v20)!=spec['required_v20_csv_sha256']:raise ValueError('Saved V20 components differ.')
    if csv_digest(ids,v21)!=spec['required_v21_csv_sha256']:raise ValueError('Saved V21 components differ.')
    if csv_digest(ids,reconstructed)!=digest(a.submission):raise ValueError('V22 independent recomposition differs.')
    if not np.array_equal(reconstructed,d['prediction']):raise ValueError('Unrounded mixture differs.')
    report=validate(a.submission,np.load(a.rowids))
    report.update(v20_components_recompose_exactly=True,v21_components_recompose_exactly=True,v22_components_recompose_exactly=True,
                  source_files_verified=len(spec['source_file_sha256']),model_and_fold_files_verified=count,
                  spec_sha256=digest(a.extension/'models/v22_spec.json'),
                  verification_scope='Independent algebraic recomposition of saved predictions, not a second complete parent inference run.')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
