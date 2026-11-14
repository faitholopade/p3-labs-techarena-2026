"""Build matched chronological tree features using V17's user-refined factors."""
import argparse,gc,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from factor_ensemble import update_features


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','extension','evidence','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);ev=np.load(a.evidence)
    names=json.loads((a.cache/'earlier_names.json').read_text());n=len(ev['labels']);raw=np.lib.format.open_memmap(a.out/'raw.npy',mode='w+',dtype=np.float32,shape=(n,len(names)))
    fold=np.empty(n//10,np.int8);seen=np.zeros(n//10,bool)
    for period in ['earlier','later']:
        x=np.load(a.cache/f'{period}_raw.npy');query=dict(np.load(a.cache/f'{period}_query.npz'))
        align=np.load(a.cache/f'{period}_alignment.npz');sel=align['selection'];rows=(sel[:,None]*10+np.arange(10)).ravel()
        assert not seen[sel].any() and json.loads((a.cache/f'{period}_names.json').read_text())==names
        update_features(x,query,a.extension/f'validation/factors/{period}_factor_users_only.pt',posterior_samples=256)
        raw[rows]=x;fold[sel]=align['fold'];seen[sel]=True;raw.flush();del x;gc.collect();print({'prepared':period,'events':len(sel)},flush=True)
    assert seen.all()
    np.save(a.out/'fold.npy',fold);(a.out/'names.json').write_text(json.dumps(names))
    (a.out/'spec.json').write_text(json.dumps({'factor_version':'v17','factor_name':'factor_users_only.pt','posterior_samples':256,'rows':n,'offset_column':names.index('posterior1.0_mix1.0_utility'),'folds_exclude_users_across_dates':True},indent=2))


if __name__=='__main__':main()
