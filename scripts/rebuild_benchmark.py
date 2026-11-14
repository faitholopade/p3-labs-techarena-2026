"""Restore and verify the shared V38 benchmark, then rebuild derived features."""
import argparse,gc,hashlib,json,os,shutil,stat,sys,tempfile,zipfile
from pathlib import Path,PurePosixPath
REPO=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(REPO/'src'),str(REPO/'scripts')]
import numpy as np
import torch

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n')

def runtime_path(root,version):
    for name in [f'v{version}','P3_Labs_v11_model_and_code' if version==11 else f'P3_Labs_v{version}_models']:
        p=Path(root)/name
        if p.is_dir():return p
    raise FileNotFoundError(f'Missing runtime v{version} in {root}')

def verify_inputs(folder,expected_metadata=None):
    if expected_metadata and sha(folder/'metadata.json')!=expected_metadata:raise ValueError('Benchmark metadata changed')
    meta=json.loads((folder/'metadata.json').read_text())
    for name,entry in meta['files'].items():
        relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:raise ValueError('Unsafe input path: '+name)
        p=folder/name
        if p.is_symlink() or sha(p)!=entry['sha256']:raise ValueError('Benchmark checksum mismatch: '+name)
    return meta

def restore(out):
    packed=REPO/'artifacts/benchmark_v38';manifest=json.loads((packed/'manifest.json').read_text())
    folder=out/'inputs'
    if folder.exists():return folder,verify_inputs(folder,manifest['metadata_sha256'])
    out.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='benchmark_restore_',dir=out) as tmp:
        tmp=Path(tmp);archive=tmp/'benchmark.zip'
        with archive.open('wb') as dest:
            for part in manifest['parts']:
                p=packed/part['path']
                if p.stat().st_size!=part['bytes'] or sha(p)!=part['sha256']:raise ValueError('Archive part mismatch: '+str(p))
                with p.open('rb') as source:shutil.copyfileobj(source,dest)
        if sha(archive)!=manifest['archive_sha256']:raise ValueError('Joined benchmark archive checksum mismatch')
        staged=tmp/'inputs';staged.mkdir()
        with zipfile.ZipFile(archive) as z:
            for info in z.infolist():
                p=PurePosixPath(info.filename)
                if p.is_absolute() or '..' in p.parts or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive path')
            if z.testzip() is not None:raise ValueError('Benchmark archive CRC failure')
            z.extractall(staged)
        meta=verify_inputs(staged,manifest['metadata_sha256']);staged.rename(folder)
    return folder,meta

def build_features(runtime,inputs,out):
    from common import load_data,subset,binary_labels
    from daily_inference import build
    from history_tree_features import extend
    from v25_exposure_memory import features as memory_features
    root=runtime_path(runtime,11);v13=runtime_path(runtime,13);v17=runtime_path(runtime,17)
    d,_,audit=load_data(root/'data');names=json.loads((inputs/'names.json').read_text());reports=[]
    for label,days in [('thursday',[1737590400]),('friday',[1737676800]),('weekend',[1737763200,1737849600])]:
        folder=out/'features'/label;folder.mkdir(parents=True,exist_ok=True)
        original=json.loads((inputs/f'original_reports/{label}.json').read_text());expected=original['feature_sha256']
        if (folder/'raw.npy').exists():
            if sha(folder/'raw.npy')!=expected:raise ValueError('Existing feature matrix differs: '+label)
            shutil.copyfile(inputs/f'alignment/{label}.npz',folder/'alignment.npz')
            shutil.copyfile(inputs/'names.json',folder/'names.json');shutil.copyfile(inputs/f'original_reports/{label}.json',folder/'report.json')
            reports.append({'cohort':label,'sha256':expected,'already_verified':True});continue
        al=dict(np.load(inputs/f'alignment/{label}.npz'));indices=al['indices']
        if not np.array_equal(binary_labels(subset(d,indices)).ravel(),al['labels']):raise ValueError('Labels misaligned')
        partial=folder/'raw.partial.npy'
        if partial.exists():raise ValueError('Unverified partial features: '+str(partial))
        dest=np.lib.format.open_memmap(partial,mode='w+',dtype=np.float32,shape=(10*len(indices),len(names)))
        for period,cut in enumerate(days):
            local=np.flatnonzero(al['period']==period) if label=='weekend' else np.arange(len(indices))
            ix=indices[local];fit=subset(d,np.flatnonzero(d['timestamp']<cut));query=subset(d,ix);query['target']=np.full(len(ix),-1)
            assert fit['timestamp'].max()<cut<=query['timestamp'].min() and query['timestamp'].max()<cut+86400
            factor=v17/f'validation/factors/{["earlier","later"][period]}_factor_users_only.pt' if label=='weekend' else inputs/f'factors/{label}.pt'
            state=torch.load(factor,map_location='cpu',weights_only=False);assert state['cut']==cut
            basis_source=v13/'validation/source_states'/str(cut) if label=='weekend' else inputs/'bases'/str(cut)
            bases=out/'states'/str(cut);bases.mkdir(parents=True,exist_ok=True)
            for p in basis_source.iterdir():
                if p.is_file():
                    target=bases/p.name
                    if not target.exists():shutil.copyfile(p,target)
                    elif sha(target)!=sha(p):raise ValueError('Basis changed: '+str(target))
            print({'stage':'features','cohort':label,'cutoff':cut,'events':len(ix)},flush=True)
            raw,base_names=build(fit,query,cut,audit['n_items'],audit['n_users'],factor,state_root=out/'states',posterior_samples=256)
            memory,memory_names=memory_features(fit,query,state['state'],cut)
            for start in range(0,len(ix),4000):
                stop=min(start+4000,len(ix));extended,en=extend(raw[start*10:stop*10],subset(query,np.arange(start,stop)),state['state'],base_names)
                assert en+memory_names==names
                rows=(local[start:stop,None]*10+np.arange(10)).ravel();dest[rows,:344]=extended;dest[rows,344:]=memory[start*10:stop*10]
            dest.flush();del raw,memory,fit,query,state;gc.collect()
        del dest;gc.collect();actual=sha(partial)
        if actual!=expected:raise ValueError(f'Feature reproduction failed for {label}: {actual} != {expected}')
        partial.rename(folder/'raw.npy');shutil.copyfile(inputs/f'alignment/{label}.npz',folder/'alignment.npz')
        shutil.copyfile(inputs/'names.json',folder/'names.json');shutil.copyfile(inputs/f'original_reports/{label}.json',folder/'report.json')
        reports.append({'cohort':label,'sha256':actual,'events':len(indices),'columns':len(names),'exact_feature_reproduction':True})
        write(out/'feature_reproduction.json',reports);print(reports[-1],flush=True)
    write(out/'feature_reproduction.json',reports)

def verify_predictions(runtime,inputs,meta,out):
    import lightgbm as lgb
    from calibration_model import predict
    from v30_tree import predict as tree_predict
    from sklearn.metrics import roc_auc_score
    controls={};checks=[]
    for label in ['thursday','friday','weekend']:
        folder=out/'features'/label;raw=np.load(folder/'raw.npy',mmap_mode='r');al=dict(np.load(folder/'alignment.npz'))
        pos=al.get('first_position',al.get('matched_position'));parts={k:np.full(len(pos)*10,np.nan) for k in meta['component_weights']}
        if label=='weekend':parts['neural32']=np.full(len(pos)*10,np.nan)
        for fold in range(3):
            test=np.flatnonzero(al['fold'][pos]==fold)
            for kind in parts:
                if kind=='tree':
                    model=lgb.Booster(model_file=str(inputs/f'folds/tree/fold_{fold}.txt'));spec=json.loads((inputs/f'folds/tree/fold_{fold}_spec.json').read_text())
                else:
                    path=runtime_path(runtime,20)/f'validation/fold_models/neural32_{fold}.pt' if kind=='neural32' else inputs/f'folds/{kind}/fold_{fold}.pt'
                    model=torch.load(path,map_location='cpu',weights_only=False)
                for start in range(0,len(test),1500):
                    events=test[start:start+1500];rows=(pos[events,None]*10+np.arange(10)).ravel();target=(events[:,None]*10+np.arange(10)).ravel()
                    block=raw[rows,:344] if kind in ['base','tree','neural32'] else raw[rows]
                    parts[kind][target]=tree_predict(model,block,spec) if kind=='tree' else predict(model,block)
        assert all(np.isfinite(v).all() for v in parts.values())
        ref=dict(np.load(inputs/('evidence/matched_v38.npz' if label=='weekend' else f'evidence/{label}_components.npz')))
        assert np.array_equal(ref['indices'],al['indices'][pos])
        mix=sum(w*parts[k] for k,w in meta['component_weights'].items())
        if label=='weekend':
            v17=np.load(runtime_path(runtime,17)/'validation/evidence.npz');assert np.array_equal(v17['indices'],ref['indices'])
            mix=mix+.43963623046875*v17['candidate']+.07177734375*parts['neural32']
        else:mix/=sum(meta['component_weights'].values())
        error=float(np.max(abs(mix-ref['candidate'])))
        if error>1e-7:raise ValueError(f'Baseline reproduction failed for {label}: {error}')
        parts.update(indices=ref['indices'],labels=ref['labels'],users=ref['users'],candidate=mix)
        if 'period' in ref:parts['period']=ref['period']
        target=out/'predictions'/f'{label}.npz';target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**parts)
        checks.append({'cohort':label,'max_abs_error':error,'auc':float(roc_auc_score(ref['labels'],mix)),'full_ensemble':label=='weekend','rows':len(mix)})
        print(checks[-1],flush=True)
    write(out/'baseline_reproduction.json',checks)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runtime',type=Path,default=REPO/'runtime');p.add_argument('--out',type=Path,required=True)
    p.add_argument('--stage',choices=['extract','features','predictions','all'],default='all');a=p.parse_args();torch.set_num_threads(3)
    inputs,meta=restore(a.out)
    if a.stage in ['features','all']:build_features(a.runtime,inputs,a.out)
    if a.stage in ['predictions','all']:verify_predictions(a.runtime,inputs,meta,a.out)
    print({'stage':a.stage,'verified_inputs':len(meta['files']),'output':str(a.out)},flush=True)

if __name__=='__main__':main()
