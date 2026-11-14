"""Verify two V34 inferences and package exactly one competition CSV."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['csv', 'recheck', 'reference', 'spec', 'report']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    digest = sha(args.csv)
    assert sha(args.recheck) == digest, 'Second inference differs'
    components_path = args.csv.with_suffix('.components.npz')
    with zipfile.ZipFile(components_path) as archive:
        assert archive.testzip() is None
    with np.load(components_path) as archive:
        components = {name: archive[name] for name in archive.files}
    frame = pd.read_csv(args.csv)
    reference = np.load(args.reference)
    assert frame.columns.tolist() == ['rowid', 'predicted_probability']
    assert len(frame) == 529390 and frame.rowid.nunique() == 529390
    assert np.array_equal(frame.rowid.to_numpy(), reference)
    assert np.array_equal(components['rowid'], reference)
    probability = frame.predicted_probability.to_numpy()
    assert np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1
    assert probability.reshape(-1, 10).sum(1).max() <= 1 + 1e-8
    spec = json.loads(args.spec.read_text())
    assert spec['version'] == 'v34'
    rebuilt = spec['v20_weight'] * components['v20']
    for head in spec['heads']:
        rebuilt = rebuilt + head['weight'] * components[head['name']]
    serialized = pd.DataFrame({'rowid': reference, 'predicted_probability': rebuilt}).to_csv(index=False, float_format='%.10g')
    assert hashlib.sha256(serialized.encode()).hexdigest() == digest
    validation = json.loads(args.csv.with_suffix('.validation.json').read_text())
    assert validation['spec_sha256'] == sha(args.spec)
    for parent in ['v27', 'v29', 'v31', 'v33']:
        assert validation['scored_parent_hashes'][parent] == spec['required_' + parent + '_csv_sha256']
    destination = args.csv.with_suffix('.csv.zip')
    temporary = destination.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        info = zipfile.ZipInfo(args.csv.name, date_time=(2026, 10, 29, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, args.csv.read_bytes())
    with zipfile.ZipFile(temporary) as archive:
        assert archive.namelist() == [args.csv.name]
        assert archive.testzip() is None
        assert hashlib.sha256(archive.read(args.csv.name)).hexdigest() == digest
    os.replace(temporary, destination)
    report = {'version': 'v34', 'rows': len(frame), 'csv_sha256': digest,
              'filename': destination.name, 'bytes': destination.stat().st_size,
              'sha256': sha(destination), 'files': 1, 'zip_crc_valid': True,
              'reference_order_matches': True, 'finite_probabilities_in_range': True,
              'second_inference_exact': True, 'component_recomposition_exact': True,
              'scored_parents_reproduced': validation['scored_parent_hashes'],
              'limits': 'Both inference runs use the checksum-verified historical V20 cache; the parent chain was not retrained.',
              'official_score': None, 'portal_uploads': 0}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
