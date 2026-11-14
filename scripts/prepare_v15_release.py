"""Assemble immutable full-data V15 weights and their validation provenance."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--v13-extension',required=True);p.add_argument('--previous',required=True)
    p.add_argument('--experiments',required=True);p.add_argument('--out',required=True);a=p.parse_args()
    old=Path(a.previous);exp=Path(a.experiments);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    (out/'models').mkdir();(out/'validation/factors').mkdir(parents=True)
    report=json.loads((exp/'assessment/selection.json').read_text());chosen=report['selection']
    members=chosen['members'];files=[];hashes={};recipes={};validation_sources={}
    for member in members:
        if member in ['original1','qmc1']:
            suffix='20261027';source=exp/'full_seed20261027';validation={period:old/f'{period}_prior05_warm2_rank05' for period in ['earlier','later']}
        elif member in ['original2','qmc2']:
            suffix='20261028';source=exp/'full_seed20261028';validation={period:old/f'{period}_prior05_warm2_rank05_seed2' for period in ['earlier','later']}
        elif member=='alt':
            suffix='alt6173';source=exp/'full_altinit';validation={period:exp/f'{period}_altinit' for period in ['earlier','later']}
        else:raise ValueError('Unknown factor member.')
        name=f'factor_{suffix}.pt';destination=out/'models'/name
        shutil.copyfile(source/'residual.pt',destination);files.append(name);hashes[name]=digest(destination)
        training=json.loads((source/'training.json').read_text())['args']
        recipes[name]={k:v for k,v in training.items() if k not in ['root','out','pool_cache','exposure','init']}
        recipes[name]['initial_factor_name']=Path(training['init']).name
        validation_sources[name]={}
        for period,folder in validation.items():
            filename=f'{period}_{name}';dest=out/'validation/factors'/filename
            shutil.copyfile(folder/'residual.pt',dest)
            validation_sources[name][period]={'file':filename,'sha256':digest(dest),'training_args':json.loads((folder/'training.json').read_text())['args']}
    for filename in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json']:
        shutil.copyfile(exp/'assessment'/filename,out/'validation'/filename)
    repo=Path(__file__).resolve().parents[1]
    sources={p.relative_to(repo).as_posix():digest(p) for p in sorted((repo/'src').glob('*.py'))}
    decision=json.loads((exp/'assessment/decision.json').read_text())
    spec={'version':'v15','created_on':'2026-10-27','required_v13_spec':json.loads((Path(a.v13_extension)/'models/v13_spec.json').read_text()),
          'factor_models':files,'factor_sha256':hashes,'factor_weights':chosen['factor_weights'],
          'posterior_samples':64,'factor_posterior_samples':chosen['factor_posterior_samples'],
          'weights':[.5,.1875,.3125],'full_training_cutoff':1737936000,'training_recipes':recipes,
          'validation_factor_sources':validation_sources,'development_result':chosen['cohorts'],
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':report['authorization'],'official_auc':None,'incumbent_official_auc':.832281,
          'source_repository':'faitholopade/p3-labs-techarena-2026','source_file_sha256':sources,
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
          'limits':'Selected on reused development cohorts; no forecast of public AUC. The general submission threshold remains unchanged.'}
    (out/'models/v15_spec.json').write_text(json.dumps(spec,indent=2))
    print(json.dumps({'output':str(out),'models':files,'mode':spec['release_mode'],'local_gain':chosen['cohorts']['combined']['gain']}),flush=True)


if __name__=='__main__':main()
