"""Freeze the selected V22 residual-tree blend, parents, and validation evidence."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn,lightgbm


def digest(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['runtime','experiments','cache','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads((a.experiments/'selection/selection.json').read_text())
    decision=json.loads((a.experiments/'assessment/decision.json').read_text())
    if chosen['mode']!='direct_blend' or chosen['parent']!='v21' or chosen['members']!=['residual_binary3']:
        raise ValueError('This release script supports only the selected, assessed V22 candidate.')
    if not chosen['eligible'] or not decision['both_dates_improve']:raise ValueError('Candidate failed selection checks.')
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/fold_models').mkdir(parents=True)
    name=chosen['members'][0];folder=a.experiments/f'full_{name}'
    training=json.loads((folder/'spec.json').read_text());screen=json.loads((a.experiments/name/'report.json').read_text())
    if training['config']!=screen['config']:raise ValueError('Full and fold training configurations differ.')
    target=a.out/f'models/{name}.txt';shutil.copyfile(folder/'full.txt',target);folds=[]
    for f in range(3):
        record={}
        for suffix in ['.txt','_spec.json']:
            path=a.out/f'validation/fold_models/{name}_{f}{suffix}'
            shutil.copyfile(a.experiments/name/f'fold_{f}{suffix}',path)
            record['model' if suffix=='.txt' else 'spec']={'file':path.relative_to(a.out).as_posix(),'sha256':digest(path)}
        folds.append(record)
    for folder_name,files in [('selection',['evidence.npz','increment_evidence.npz','selection.json','grid.json']),
                              ('assessment',['report.json','paired_bootstrap.json','decision.json']),
                              ('increment_assessment',['report.json','paired_bootstrap.json','decision.json'])]:
        dst=a.out/'validation'/folder_name;dst.mkdir()
        for file in files:shutil.copyfile(a.experiments/folder_name/file,dst/file)
    repo=Path(__file__).resolve().parents[1]
    spec={'version':'v22','created_on':'2026-10-28','mode':chosen['mode'],'parent':chosen['parent'],
          'weight':chosen['weight'],'formula':'(1-weight)*v21 + weight*residual_binary3',
          'required_parent_spec_sha256':{f'v{v}':digest(a.runtime/f'P3_Labs_v{v}_models/models/v{v}_spec.json') for v in [20,21]},
          'required_v20_csv_sha256':'3e82021c98bec56bce493b3a56a335b8778e124388293ae5035496bd358c01c5',
          'required_v21_csv_sha256':'7878542d77a88774357da602231cee0dc01de23936559fdee1a47453d7cf342c',
          'cache_file_sha256':{name:digest(a.cache/name) for name in ['raw.npy','names.json','components.npz','reproduction.json']},
          'heads':[{'name':name,'kind':'residual_binary_tree','file':target.name,'sha256':digest(target),
                    'training_spec':training,'validation_fold_models':folds}],
          'selection':chosen,'development_result':json.loads((a.experiments/'assessment/report.json').read_text()),
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':chosen['authorization'],'official_auc':None,'incumbent_official_auc':.832548,
          'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{f.relative_to(repo).as_posix():digest(f) for folder in ['src','scripts'] for f in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__,'lightgbm':lightgbm.__version__},
          'limits':'Adaptive selection on reused dates. Conditional bootstrap excludes selection and distribution shift. Long-gap diagnostic reuses outcomes and measures calibration heads, not the full parent.'}
    (a.out/'models/v22_spec.json').write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({'weight':spec['weight'],'members':chosen['members'],'release_mode':spec['release_mode'],'training_events':training['training_events']}),flush=True)


if __name__=='__main__':main()
