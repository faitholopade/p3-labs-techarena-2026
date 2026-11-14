"""Package selected V17 factors, chronological evidence, and exact V16 provenance."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np,torch,scipy,sklearn


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['v16-extension','experiments','selection','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();chosen=json.loads(a.selection.read_text())
    parent=json.loads((a.v16_extension/'models/v16_spec.json').read_text())
    a.out.mkdir(parents=True,exist_ok=False);(a.out/'models').mkdir();(a.out/'validation/factors').mkdir(parents=True)
    files=[];hashes={};recipes={};validation={}
    for member in chosen['members']:
        name=f'factor_{member}.pt';files.append(name);folder=a.experiments/f'full_{member}'
        source=folder/'residual.pt';checkpoint=torch.load(source,map_location='cpu',weights_only=False)
        if checkpoint['cut']!=1737936000:raise ValueError('Full-data cutoff mismatch.')
        dest=a.out/'models'/name;shutil.copyfile(source,dest);hashes[name]=digest(dest)
        if (folder/'training.json').exists():
            args=json.loads((folder/'training.json').read_text())['args']
            recipe={k:v for k,v in args.items() if k not in ['root','out','pool_cache','exposure','init']}
            recipe['initial_factor_name']=Path(args['init']).name;recipe['script']='src/refine_exposure_prior.py'
        else:
            recipe=json.loads((folder/'adaptation.json').read_text());recipe['script']='src/adapt_query_history.py'
            recipe['initial_factor_name']='factor_antithetic_seed2.pt';recipe['cut']=1737936000
            recipe['query_source']='Supplied evaluation histories and candidate item IDs; no outcomes.'
        recipes[name]=recipe;validation[name]={}
        for period,cut in [('earlier',1737763200),('later',1737849600)]:
            source=a.experiments/f'{period}_{member}/residual.pt'
            if torch.load(source,map_location='cpu',weights_only=False)['cut']!=cut:raise ValueError('Validation cutoff mismatch.')
            vname=f'{period}_{name}';dest=a.out/'validation/factors'/vname;shutil.copyfile(source,dest)
            validation[name][period]={'file':vname,'sha256':digest(dest),'cut':cut}
    for name in ['evidence.npz','report.json','paired_bootstrap.json','decision.json','selection.json']:
        shutil.copyfile(a.selection.parent/name,a.out/'validation'/name)
    repo=Path(__file__).resolve().parents[1];decision=json.loads((a.selection.parent/'decision.json').read_text())
    spec={'version':'v17','created_on':'2026-10-27','required_v16_spec':parent,
          'required_v16_csv_sha256':'e8bb1b30c81648dbd5dc4983bcbf0305879ddf7ee9896319ab3cb26d9e93b6a2',
          'factor_models':files,'factor_sha256':hashes,'factor_weights':chosen['factor_weights'],
          'factor_posterior_samples':chosen['factor_posterior_samples'],'weights':chosen['weights'],
          'weight_order':['v16','v15','v8','new_sequence'],'full_training_cutoff':1737936000,
          'training_recipes':recipes,'validation_factor_sources':validation,
          'development_result':json.loads((a.selection.parent/'report.json').read_text()),
          'release_mode':'normal_gate' if decision['normal_quality_gate'] else 'user_authorized_exploratory',
          'authorization':chosen['authorization'],'official_auc':None,'incumbent_official_auc':.832391,
          'source_repository':'faitholopade/p3-labs-techarena-2026',
          'source_file_sha256':{p.relative_to(repo).as_posix():digest(p) for folder in ['src','scripts'] for p in sorted((repo/folder).glob('*.py'))},
          'environment':{'numpy':np.__version__,'torch':torch.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},
          'limits':'Reused development cohorts and model/blend selection; no forecast of public score. Normal threshold remains unchanged.'}
    (a.out/'models/v17_spec.json').write_text(json.dumps(spec,indent=2))
    print(json.dumps({'models':files,'mode':spec['release_mode'],'weights':spec['weights']}),flush=True)


if __name__=='__main__':main()
