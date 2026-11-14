"""Run unchanged V65 inference and retain its derived evaluation feature matrix."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
import numpy as np
import infer_after_v63

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--capture',type=Path,required=True)
    args,remaining=parser.parse_known_args();args.capture.mkdir(parents=True,exist_ok=True)
    original=infer_after_v63.head_predict
    def predict(spec,raw):
        if raw.shape==(529390,388) and not (args.capture/'memory.npy').exists():
            np.save(args.capture/'memory.npy',raw)
            (args.capture/'names.json').write_text(json.dumps(spec['names'])+'\n')
        return original(spec,raw)
    infer_after_v63.head_predict=predict
    sys.argv=[str(ROOT/'src/infer_after_v63.py')]+remaining
    infer_after_v63.main()
if __name__=='__main__':main()
