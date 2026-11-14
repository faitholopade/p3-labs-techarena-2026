"""Package exactly one CSV after two independent inference outputs agree."""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
import numpy as np
import pandas as pd
from validate_submission import validate


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['first', 'repeat', 'runtime', 'spec', 'out', 'report']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--previous', type=Path)
    args = parser.parse_args(); spec = json.loads(args.spec.read_text()); version = spec['version']
    assert args.first.name == f'P3_Labs_submission_{version}.csv'
    assert args.out.name == args.first.name + '.zip'
    first_sha = sha(args.first); assert first_sha == sha(args.repeat), 'Inference outputs differ.'
    root = args.runtime / 'P3_Labs_v11_model_and_code'
    if not root.is_dir(): root = args.runtime / 'v11'
    reference = np.load(root / 'data/evaluation_rowids.npy')
    checked = validate(args.first, reference)
    assert checked['valid'] and checked['reference_order_matches'] and checked['rows'] == 529390
    inference = []
    for path in [args.first, args.repeat]:
        record = json.loads(path.with_suffix('.validation.json').read_text())
        assert record['csv_sha256'] == first_sha and record['spec_sha256'] == sha(args.spec)
        assert record['version'] == version and record['valid']
        inference.append(record)
    data = pd.read_csv(args.first)
    assert data.columns.tolist() == ['rowid', 'predicted_probability']
    values = data.predicted_probability.to_numpy()
    assert np.isfinite(values).all() and values.min() >= 0 and values.max() <= 1
    maximum_mass = float(values.reshape(-1, 10).sum(1).max()); assert maximum_mass <= 1 + 1e-8
    args.out.parent.mkdir(parents=True, exist_ok=True)
    assert not args.out.exists(), 'Refusing to overwrite an existing archive.'
    info = zipfile.ZipInfo(args.first.name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3; info.external_attr = 0o644 << 16
    with zipfile.ZipFile(args.out, 'w') as archive:
        archive.writestr(info, args.first.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    with zipfile.ZipFile(args.out) as archive:
        assert archive.namelist() == [args.first.name] and archive.testzip() is None
        assert hashlib.sha256(archive.read(args.first.name)).hexdigest() == first_sha
        assert archive.getinfo(args.first.name).file_size == args.first.stat().st_size
    report = dict(version=version, csv_filename=args.first.name, csv_sha256=first_sha,
                  csv_bytes=args.first.stat().st_size, zip_filename=args.out.name,
                  zip_sha256=sha(args.out), zip_bytes=args.out.stat().st_size, rows=529390,
                  unique_rowids=True, reference_order_matches=True, finite_unit_probabilities=True,
                  minimum_probability=float(values.min()), maximum_probability=float(values.max()),
                  max_event_probability=maximum_mass, zip_contains_exactly_one_csv=True,
                  zip_crc_valid=True, zip_member_matches_verified_csv=True,
                  two_inference_runs_byte_identical=True, scored_parent_hashes=inference[0]['scored_parent_hashes'],
                  spec_sha256=sha(args.spec), inference_source_sha256=inference[0]['source_sha256'],
                  packaging_source_sha256=sha(Path(__file__)),
                  normal_quality_gate=spec['normal_quality_gate'], release_mode=spec['release_mode'],
                  official_auc=None, portal_uploads=0,
                  scope='Two saved-model inference runs; every parent listed in scored_parent_hashes reproduced exactly. This does not retrain historical factors.',
                  inference_modes=[record['inference_mode'] for record in inference])
    if args.previous:
        old = pd.read_csv(args.previous)
        assert np.array_equal(old.rowid.to_numpy(), data.rowid.to_numpy())
        delta = values - old.predicted_probability.to_numpy()
        report.update(changed_rows_vs_incumbent=int(np.count_nonzero(delta)),
                      max_abs_probability_change_vs_incumbent=float(np.max(np.abs(delta))))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__': main()
