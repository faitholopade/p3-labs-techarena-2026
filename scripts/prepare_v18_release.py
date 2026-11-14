"""Package selected tree heads and their exact V17 and validation provenance."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn,lightgbm


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v17-extension','experiments','selection','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text());parent=json.loads((a.v17_extension/'models/v17_spec.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/fold_models').mkdir(parents=True)
    trees=[]
    for member in chosen['members']:
        folder=a.experiments/f'full_{member}';dest=a.out/'models'/f'{member}.txt';shutil.copyfile(folder/'tree.txt',dest)
        training=json.loads((folder/'spec.json').read_text());folds=[]
        for fold in range(3):
            target=a.out/f'validation/fold_models/{member}_{fold}.txt';shutil.copyfile(a.experiments/member/f'fold_{fold}.txt',target)
            folds.append({'file':target.relative_to(a.out).as_posix(),'sha256':digest(target)})
        trees.append({'name':member,'file':dest.name,'sha256':digest(dest),'training_spec':training,'validation_fold_models':folds})
    for name in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json','grid.json']:
        shutil.copyfile(a.selection.parent/name,a.out/'validation'/name)
    repo=Path(__file__).resolve().parents[1];decision=json.loads((a.selection.parent/'decision.json').read_text())
    spec={'version':'v18','created_on':'2026-10-27','required_v17_spec':parent,
          'required_v17_csv_sha256':'8b500ac995607f253b9a817cbdeae0dbd3b7d7d529728e9851c04901291b236c',
          'tree_models':trees,'weights':chosen['weights'],'weight_order':['v17']+chosen['members'],
          'calibration_rows':703680,'calibration_events':70368,'feature_cutoffs':[1737763200,1737849600],
          'inference_factor_cutoff':1737936000,'full_head_fit':'All matched calibration rows; parent factors use all pre-evaluation training events.',
          'development_result':json.loads((a.selection.parent/'report.json').read_text()),
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':chosen['authorization'],'official_auc':None,'incumbent_official_auc':.832425,
          'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{f.relative_to(repo).as_posix():digest(f) for folder in ['src','scripts'] for f in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__},
          'limits':'Reused development cohorts and blend selection. Public/private distribution shift and selection uncertainty are not covered by the paired bootstrap.'}
    (a.out/'models/v18_spec.json').write_text(json.dumps(spec,indent=2));print(json.dumps({'members':chosen['members'],'weights':chosen['weights'],'release_mode':spec['release_mode']}),flush=True)


if __name__=='__main__':main()
