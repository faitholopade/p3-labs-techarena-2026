"""Package selected V16 factors with their immutable baseline and validation provenance."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v15-extension','experiments','selection','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text())
    v15=json.loads((a.v15_extension/'models/v15_spec.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/factors').mkdir(parents=True)
    files=[];hashes={};recipes={};validation={}
    for member in chosen['members']:
        name=f'factor_{member}.pt';dest=a.out/'models'/name;files.append(name)
        if member=='incumbent_alt':
            source=a.v15_extension/'models/factor_alt6173.pt';recipe=v15['training_recipes']['factor_alt6173.pt']
            pairs={period:a.v15_extension/f'validation/factors/{period}_factor_alt6173.pt' for period in ['earlier','later']}
        else:
            folder=a.experiments/f'full_{member}';source=folder/'residual.pt'
            args=json.loads((folder/'training.json').read_text())['args']
            recipe={k:v for k,v in args.items() if k not in ['root','out','pool_cache','exposure','init']};recipe['initial_factor_name']=Path(args['init']).name
            pairs={period:a.experiments/f'{period}_{member}/residual.pt' for period in ['earlier','later']}
        checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        if checkpoint['cut']!=1737936000:raise ValueError('Full-data cutoff mismatch.')
        shutil.copyfile(source,dest);hashes[name]=digest(dest);recipes[name]=recipe;validation[name]={}
        for period,source in pairs.items():
            cut=1737763200 if period=='earlier' else 1737849600
            if torch.load(source,map_location='cpu',weights_only=False)['cut']!=cut:raise ValueError('Validation cutoff mismatch.')
            vname=f'{period}_{name}';dest=a.out/'validation/factors'/vname;shutil.copyfile(source,dest)
            validation[name][period]={'file':vname,'sha256':digest(dest),'cut':cut}
    for name in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json']:
        shutil.copyfile(a.selection.parent/name,a.out/'validation'/name)
    repo=Path(__file__).resolve().parents[1]
    decision=json.loads((a.selection.parent/'decision.json').read_text())
    spec={'version':'v16','created_on':'2026-10-27','required_v15_spec':v15,
          'required_v15_csv_sha256':'5beff1139b0e520714ee513846ba6f62d3de0a2f83dbae48c8e789abc99c2049',
          'factor_models':files,'factor_sha256':hashes,'factor_weights':chosen['factor_weights'],
          'factor_posterior_samples':chosen['factor_posterior_samples'],'weights':chosen['weights'],
          'weight_order':['v15','v12','v8','new_sequence'],'full_training_cutoff':1737936000,
          'training_recipes':recipes,'validation_factor_sources':validation,
          'development_result':json.loads((a.selection.parent/'report.json').read_text()),
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':chosen['authorization'],'official_auc':None,'incumbent_official_auc':.832366,
          'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{p.relative_to(repo).as_posix():digest(p) for folder in ['src','scripts'] for p in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
          'limits':'Reused development cohorts and model/blend selection; no forecast of public score. Normal threshold remains unchanged.'}
    (a.out/'models/v16_spec.json').write_text(json.dumps(spec,indent=2))
    print(json.dumps({'models':files,'mode':spec['release_mode'],'weights':spec['weights']}),flush=True)


if __name__=='__main__':main()
