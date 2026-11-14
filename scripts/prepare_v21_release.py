"""Package first-event calibration heads and matched validation evidence."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn,lightgbm


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ['v20-extension','experiments','selection','out']:p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text());decision=json.loads((a.selection.parent/'decision.json').read_text())
    if not decision['both_dates_improve'] or chosen['long_gap_calibration_subensemble_gain']<=0:raise ValueError('No eligible candidate.')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/fold_models').mkdir(parents=True);heads=[]
    for name in chosen['members']:
        folder=a.experiments/f'full_{name}';training=json.loads((folder/'spec.json').read_text());validation=json.loads((a.experiments/name/'report.json').read_text())
        if training['config']!=validation['config']:raise ValueError('Training configurations differ.')
        dst=a.out/f'models/{name}.pt';shutil.copyfile(folder/'model.pt',dst);folds=[]
        for f in range(3):
            path=a.out/f'validation/fold_models/{name}_{f}.pt';shutil.copyfile(a.experiments/name/f'fold_{f}.pt',path);folds.append({'file':path.relative_to(a.out).as_posix(),'sha256':digest(path)})
        heads.append({'name':name,'kind':'neural','file':dst.name,'sha256':digest(dst),'training_spec':training,'validation_fold_models':folds})
    for name in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json','grid.json']:
        shutil.copyfile(a.selection.parent/name,a.out/'validation'/name)
    repo=Path(__file__).resolve().parents[1];feature_spec=json.loads((a.experiments/'features/spec.json').read_text())
    spec={'version':'v21','created_on':'2026-10-28','required_v20_spec':json.loads((a.v20_extension/'models/v20_spec.json').read_text()),
          'required_v20_csv_sha256':'3e82021c98bec56bce493b3a56a335b8778e124388293ae5035496bd358c01c5',
          'replacement_weight':chosen['effective_weight'],'formula':'v20 + replacement_weight * (mean(new heads) - v20_neural32)',
          'heads':heads,'feature_spec':feature_spec,'selection':chosen,'development_result':json.loads((a.selection.parent/'report.json').read_text()),
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory','authorization':chosen['authorization'],
          'official_auc':None,'incumbent_official_auc':.832548,'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{f.relative_to(repo).as_posix():digest(f) for folder in ['src','scripts'] for f in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__},
          'limits':'Selected on repeatedly reused development dates. Conditional bootstrap excludes selection and distribution shift. Longer-gap check reuses outcomes and compares calibration heads only; evaluation gaps beyond two days remain unsupported.'}
    (a.out/'models/v21_spec.json').write_text(json.dumps(spec,indent=2));print({'members':chosen['members'],'replacement_weight':chosen['effective_weight'],'release_mode':spec['release_mode']},flush=True)


if __name__=='__main__':main()
