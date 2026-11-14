"""Package expanded-data calibration heads with immutable V19 provenance."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn,lightgbm


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v19-extension','experiments','selection','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text());decision=json.loads((a.selection.parent/'decision.json').read_text())
    if not decision['both_dates_improve']:raise ValueError('Release requires both dates to improve.')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/fold_models').mkdir(parents=True);heads=[]
    for member in chosen['members']:
        folder=a.experiments/f'full_{member}';training=json.loads((folder/'spec.json').read_text());validation=json.loads((a.experiments/member/'report.json').read_text())
        if training['config']!=validation['config']:raise ValueError('Full and validation configurations differ.')
        kind='tree' if member=='expanded700' else 'neural';suffix='.txt' if kind=='tree' else '.pt';filename='tree.txt' if kind=='tree' else 'model.pt'
        target=a.out/'models'/f'{member}{suffix}';shutil.copyfile(folder/filename,target);folds=[]
        for f in range(3):
            dst=a.out/f'validation/fold_models/{member}_{f}{suffix}';shutil.copyfile(a.experiments/member/f'fold_{f}{suffix}',dst)
            folds.append({'file':dst.relative_to(a.out).as_posix(),'sha256':digest(dst)})
        heads.append({'name':member,'kind':kind,'file':target.name,'sha256':digest(target),'training_spec':training,'validation_fold_models':folds})
    for filename in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json','grid.json']:
        shutil.copyfile(a.selection.parent/filename,a.out/'validation'/filename)
    repo=Path(__file__).resolve().parents[1]
    spec={'version':'v20','created_on':'2026-10-27','required_v19_spec':json.loads((a.v19_extension/'models/v19_spec.json').read_text()),
          'required_v19_csv_sha256':'5e411f6d96218795f616722d3ebd123b43646bfb74a47a4fd12d78210b891efe',
          'weights':chosen['weights'],'weight_order':chosen['weight_order'],'heads':heads,'calibration_rows':1964040,'calibration_events':196404,
          'feature_cutoffs':[1737763200,1737849600],'inference_factor_cutoff':1737936000,
          'full_head_fit':'All calibration events from both complete days; validation heads exclude held-out users across dates.',
          'development_result':json.loads((a.selection.parent/'report.json').read_text()),'selection':chosen,
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory','authorization':chosen['authorization'],
          'official_auc':None,'incumbent_official_auc':.832540,'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{f.relative_to(repo).as_posix():digest(f) for folder in ['src','scripts'] for f in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__},
          'limits':'Repeated development reuse and selection; conditional bootstrap excludes selection and distribution-shift uncertainty.'}
    (a.out/'models/v20_spec.json').write_text(json.dumps(spec,indent=2));print(json.dumps({'members':chosen['members'],'weights':chosen['weights'],'release_mode':spec['release_mode']}),flush=True)


if __name__=='__main__':main()
