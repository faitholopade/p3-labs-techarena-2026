"""Extend matched V18 features without using labels or future interactions."""
import argparse,json,sys,gc
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,torch
from history_tree_features import extend


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['cache','features','extension','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);old=np.load(a.features/'raw.npy',mmap_mode='r');names=json.loads((a.features/'names.json').read_text());result=None
    seen=np.zeros(len(old)//10,bool)
    for period in ['earlier','later']:
        align=np.load(a.cache/f'{period}_alignment.npz');selection=align['selection'];rows=(selection[:,None]*10+np.arange(10)).ravel()
        if seen[selection].any():raise ValueError('Overlapping query cohorts.')
        events=dict(np.load(a.cache/f'{period}_query.npz'));state=torch.load(a.extension/f'validation/factors/{period}_factor_users_only.pt',map_location='cpu',weights_only=False)['state']
        x,extended_names=extend(old[rows],events,state,names)
        if result is None:
            result=np.lib.format.open_memmap(a.out/'raw.npy',mode='w+',dtype=np.float32,shape=(len(old),x.shape[1]));all_names=extended_names
        if extended_names!=all_names:raise ValueError('Schemas differ.')
        result[rows]=x;seen[selection]=True;result.flush();del x,state;gc.collect();print({'period':period,'events':len(selection),'columns':len(all_names)},flush=True)
    assert seen.all()
    np.save(a.out/'fold.npy',np.load(a.features/'fold.npy'))
    (a.out/'names.json').write_text(json.dumps(all_names))
    spec=json.loads((a.features/'spec.json').read_text());spec.update({'extra_features':'history_tree_features.extend','columns':len(all_names),'outcome_free_extension':True})
    (a.out/'spec.json').write_text(json.dumps(spec,indent=2))


if __name__=='__main__':main()
