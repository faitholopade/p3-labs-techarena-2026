"""Verify and import a known private P3 Labs artifact bundle, without network access."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--version', default='v9')
    parser.add_argument('--destination', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'artifacts/manifest.json').read_text())
    spec = manifest[args.version]
    if spec.get('storage') == 'private_git':
        raise SystemExit(f"This small checkpoint is already included at {spec['path']}; no archive import is needed.")
    with args.archive.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    if digest != spec['sha256']:
        raise SystemExit('Archive SHA-256 mismatch; no files were imported.')
    destination = (args.destination or ROOT / 'runtime' / args.version).resolve()
    prefix = PurePosixPath(spec['prefix'])
    selected = []
    with zipfile.ZipFile(args.archive) as archive:
        for info in archive.infolist():
            path = PurePosixPath(info.filename)
            if path.is_absolute() or '..' in path.parts or '\\' in info.filename:
                raise SystemExit('Unsafe archive path.')
            if stat.S_ISLNK(info.external_attr >> 16):
                raise SystemExit('Archive symlinks are not supported.')
            if not path.is_relative_to(prefix):
                raise SystemExit('Unexpected archive root.')
            relative = path.relative_to(prefix)
            if not relative.parts or relative.parts[0] not in {'data', 'models', 'validation'}:
                continue
            if not info.is_dir():
                selected.append((info, relative))
        if args.check_only:
            print(json.dumps({'sha256': digest, 'files_to_import': len(selected), 'verified': True}))
            return
        if destination.exists():
            raise SystemExit(f'Destination already exists: {destination}; use a new directory.')
        destination.mkdir(parents=True)
        for info, relative in selected:
            target = destination.joinpath(*relative.parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open('xb') as output:
                shutil.copyfileobj(source, output)
    print(json.dumps({'destination': str(destination), 'files_imported': len(selected), 'sha256': digest}))


if __name__ == '__main__':
    main()

