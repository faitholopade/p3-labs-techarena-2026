"""Package selected tree heads with immutable V18 provenance and evidence."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn,lightgbm


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v18-extension','v17-extension','experiments','selection','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text());decision=json.loads((a.selection.parent/'decision.json').read_text());parent=json.loads((a.v18_extension/'models/v18_spec.json').read_text())
    if not decision['both_dates_improve']:raise ValueError('Release requires both dates to improve.')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/fold_models').mkdir(parents=True);trees=[]
    for member in chosen['members']:
        folder=a.experiments/f'full_{member}';training=json.loads((folder/'spec.json').read_text());validation=json.loads((a.experiments/member/'report.json').read_text())
        if training['config']!=validation['config'] or training['names']!=validation['training_spec']['names']:raise ValueError('Full and validation configurations differ.')
        target=a.out/'models'/f'{member}.txt';shutil.copyfile(folder/'tree.txt',target);folds=[]
        for f in range(3):
            dst=a.out/f'validation/fold_models/{member}_{f}.txt';shutil.copyfile(a.experiments/member/f'fold_{f}.txt',dst);folds.append({'file':dst.relative_to(a.out).as_posix(),'sha256':digest(dst)})
        trees.append({'name':member,'file':target.name,'sha256':digest(target),'training_spec':training,'validation_fold_models':folds})
    for filename in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json','grid.json']:
        shutil.copyfile(a.selection.parent/filename,a.out/'validation'/filename)
    repo=Path(__file__).resolve().parents[1]
    spec={'version':'v19','created_on':'2026-10-27','required_v18_spec':parent,
          'required_v18_csv_sha256':'6332a08849683970bd2d597ba35c78c936a27c5dbc4bdadda54a2dd220ed8206',
          'required_feature_factor_sha256':digest(a.v17_extension/'models/factor_users_only.pt'),
          'weights':chosen['weights'],'weight_order':chosen['weight_order'],'tree_models':trees,
          'calibration_rows':703680,'calibration_events':70368,'feature_cutoffs':[1737763200,1737849600],
          'inference_factor_cutoff':1737936000,'full_head_fit':'All matched calibration rows. Parent factors use all pre-evaluation training events.',
          'development_result':json.loads((a.selection.parent/'report.json').read_text()),'selection':chosen,
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':chosen['authorization'],'official_auc':None,'incumbent_official_auc':.832503,
          'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{f.relative_to(repo).as_posix():digest(f) for folder in ['src','scripts'] for f in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__},
          'limits':'Repeated development reuse and model selection; conditional paired bootstrap does not cover selection or distribution shift.'}
    (a.out/'models/v19_spec.json').write_text(json.dumps(spec,indent=2));print(json.dumps({'members':chosen['members'],'weights':chosen['weights'],'release_mode':spec['release_mode']}),flush=True)


if __name__=='__main__':main()
