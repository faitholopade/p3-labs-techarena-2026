"""Fit the two fixed V25/V26 heads on the complete weekend calibration pool."""
import argparse,json,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'src'));sys.path.insert(0,str(REPO/'scripts'))
import numpy as np
import torch
from daily_training import train
from backtest_v24_factor import sha256,write_json

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--features',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--family',choices=['v25','v26'],required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    raw=np.load(a.features/'raw.npy',mmap_mode='r');al=dict(np.load(a.features/'alignment.npz'));names=json.loads((a.features/'names.json').read_text())
    columns=len(names) if a.family=='v25' else 344
    config={'epochs':20,'width':32,'rank_weight':.5,'threads':3,'architecture':'plain','offset_name':'posterior1.0_mix1.0_utility'}
    plan={'family':a.family,'seeds':[20261029,20261030],'configuration':config,'columns':columns,'training_events':len(al['indices']),
          'feature_report_sha256':sha256(a.features/'report.json'),'source_sha256':sha256(Path(__file__))}
    path=a.out/'plan.json'
    if path.exists() and json.loads(path.read_text())!=plan:raise ValueError('Fixed full-fit plan changed.')
    write_json(path,plan);records=[]
    for seed in plan['seeds']:
        path=a.out/f'head_{seed}.pt';done=a.out/f'head_{seed}_completed.json'
        if done.exists():
            record=json.loads(done.read_text());assert record['sha256']==sha256(path)
        else:
            if path.exists():raise ValueError('Unverified partial full fit.')
            model=train([{'x':raw[:,:columns],'labels':al['labels'],'names':np.array(names[:columns])}],
                        [np.ones(len(al['indices']),bool)],seed=seed,**config)
            torch.save(model,path);record={'file':path.name,'sha256':sha256(path),'seed':seed,'training_events':model['train_events']};write_json(done,record)
        records.append(record)
    write_json(a.out/'report.json',{'plan':plan,'models':records})

if __name__=='__main__':main()
