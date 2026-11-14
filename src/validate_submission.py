"""Check the exact official Phase 1 submission contract against the example IDs."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

def validate(path,expected):
    frame=pd.read_csv(path,dtype={'rowid':str})
    if list(frame.columns)!=['rowid','predicted_probability']:
        raise ValueError('Incorrect columns or column order')
    expected=np.asarray(expected).astype(str)
    if len(frame)!=len(expected):raise ValueError('Incorrect number of rows')
    if len(np.unique(expected))!=len(expected):raise ValueError('Reference IDs are duplicated')
    if frame.rowid.isna().any() or frame.rowid.duplicated().any():raise ValueError('Missing or duplicate row IDs')
    if set(frame.rowid)!=set(expected):raise ValueError('Missing or unknown row IDs')
    p=pd.to_numeric(frame.predicted_probability,errors='raise').to_numpy()
    if not np.isfinite(p).all() or not ((p>=0)&(p<=1)).all():raise ValueError('Probabilities must be finite and in [0,1]')
    return dict(rows=len(frame),unique_rowids=frame.rowid.nunique(),reference_order_matches=bool(np.array_equal(frame.rowid.to_numpy(),expected)),min_probability=float(p.min()),max_probability=float(p.max()),mean_probability=float(p.mean()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),valid=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('submission');parser.add_argument('--example');parser.add_argument('--rowids');args=parser.parse_args()
    if bool(args.example)==bool(args.rowids):parser.error('Supply exactly one of --example or --rowids')
    ids=pd.read_csv(args.example,dtype={'rowid':str}).rowid.to_numpy() if args.example else np.load(args.rowids)
    print(json.dumps(validate(args.submission,ids),indent=2))

if __name__=='__main__':main()
