"""Preserve V37 research checkpoints; repository-owned files remain in Git."""
import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['research', 'repo', 'out', 'report']:
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--components', type=Path)
    a = p.parse_args()
    repo_files = list((a.repo / 'results').glob('*.json'))
    repo_files += list((a.repo / 'artifacts/v37').glob('*'))
    known = {sha(path): str(path.relative_to(a.repo)) for path in repo_files if path.is_file()}
    included, external = [], []
    skip = {'library_save.json', 'research_archive.json'}
    for path in sorted(a.research.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(a.research)
        if any(part.startswith('.') or part in {'__pycache__', 'recheck', 'pools'} for part in rel.parts):
            continue
        if path.name in skip or path.suffix in {'.pyc', '.partial', '.csv', '.zip'}:
            continue
        digest = sha(path)
        record = {'path': str(Path('research_v37') / rel), 'bytes': path.stat().st_size, 'sha256': digest}
        if digest in known:
            external.append(dict(record, repository_path=known[digest]))
        else:
            included.append((path, record))
    if a.components:
        included.append((a.components, {'path': 'research_v37/inference/' + a.components.name,
                        'bytes': a.components.stat().st_size, 'sha256': sha(a.components)}))
    index = {'repository': 'faitholopade/p3-labs-techarena-2026',
             'purpose': 'Research backup; never upload to the competition portal.',
             'included': [record for _, record in included], 'repository_backed': external,
             'prerequisites': ['Historical private runtime archives V11 through V20',
                               'V25-V28, V29, V30-V31 and V33 research archives',
                               'Private Git repository source, release specifications and full heads'],
             'omissions': 'Derived raw feature matrices and duplicate evaluation outputs are reproducible and excluded.'}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = a.out.with_suffix('.zip.partial')
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, record in included:
            archive.write(path, record['path'])
        archive.writestr('checkpoint_index.json', json.dumps(index, indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(included) + 1
        for _, record in included:
            assert hashlib.sha256(archive.read(record['path'])).hexdigest() == record['sha256']
    os.replace(temporary, a.out)
    report = {'filename': a.out.name, 'bytes': a.out.stat().st_size, 'sha256': sha(a.out),
              'files': len(included) + 1, 'zip_crc_valid': True, 'member_hashes_verified': True,
              'excluded_repo_backed_files': len(external), 'purpose': index['purpose']}
    a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
