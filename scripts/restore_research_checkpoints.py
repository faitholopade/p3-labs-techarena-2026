"""Restore the compact October 30 research fits with per-file verification."""
import argparse
import hashlib
import json
import shutil
import stat
import tempfile
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[1]


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='New directory; never overwrites an existing research run.')
    parser.add_argument('--bundle', choices=['research_2026_10_30', 'research_v44', 'research_v45'],
                        default='research_2026_10_30')
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError('Choose a new output directory: ' + str(args.out))
    folder = REPO / 'artifacts' / args.bundle
    manifest = json.loads((folder / 'manifest.json').read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='research_restore_', dir=args.out.parent) as temporary:
        temporary = Path(temporary)
        archive = temporary / 'checkpoints.zip'
        with archive.open('wb') as target:
            for entry in manifest['parts']:
                path = folder / entry['path']
                assert path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256']
                with path.open('rb') as source:
                    shutil.copyfileobj(source, target)
        assert sha(archive) == manifest['archive_sha256']
        staged = temporary / 'restored'
        staged.mkdir()
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                name = PurePosixPath(entry.filename)
                assert not name.is_absolute() and '..' not in name.parts
                assert not stat.S_ISLNK(entry.external_attr >> 16)
            assert source.testzip() is None
            source.extractall(staged)
        for name, expected in manifest['files_sha256'].items():
            assert sha(staged / name) == expected, name
        staged.rename(args.out)
    print(json.dumps({'restored': str(args.out), 'verified_files': len(manifest['files_sha256']),
                      'scope': 'Saved calibration fits and reports. Derived prediction arrays regenerate from the shared benchmark.'}))


if __name__ == '__main__':
    main()
