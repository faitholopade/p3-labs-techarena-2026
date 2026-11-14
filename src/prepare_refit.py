"""Prepare the full-training initialization paths used by the research trainers."""
from pathlib import Path
import argparse,shutil
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',default='.');a=ap.parse_args();root=Path(a.root).resolve();folder=root/'v6/states/1737936000';folder.mkdir(parents=True,exist_ok=True)
    for source,name in [('preference_init.npy','p0_u0.5_i0.5_r5.npy'),('exposure_basis.npz','exposure.npz')]:shutil.copyfile(root/'data'/source,folder/name)
    print('Full-training initialization is ready.')
if __name__=='__main__':main()
