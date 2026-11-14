"""Back up research checkpoints while keeping repository-owned files in Git."""
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
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['v30', 'v31', 'repo', 'components', 'out', 'report']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    repository_files = list((args.repo / 'results').glob('*.json'))
    repository_files += [args.repo / 'artifacts/v31/spec.json', args.repo / 'artifacts/v27/pooled_head.pt', args.repo / 'artifacts/v29/head.pt']
    by_hash = {sha(path): str(path.relative_to(args.repo)) for path in repository_files}
    included, external = [], []
    skip_names = {'storage_cleanup.json', 'library_save.json', 'research_archive.json'}
    for root, prefix in [(args.v30, 'research_v30'), (args.v31, 'research_v31')]:
        for path in sorted(root.rglob('*')):
            if not path.is_file():
                continue
            relative = path.relative_to(root)
            if any(part.startswith('.') or part == '__pycache__' for part in relative.parts):
                continue
            if path.name in skip_names or path.suffix in {'.csv', '.pyc', '.partial'}:
                continue
            if 'recheck' in relative.parts and path.suffix == '.npz':
                continue
            digest = sha(path)
            record = {'path': str(Path(prefix) / relative), 'bytes': path.stat().st_size, 'sha256': digest}
            if digest in by_hash:
                external.append(dict(record, repository_path=by_hash[digest]))
            else:
                included.append((path, record))
    included.append((args.components, {'path': 'research_v31/inference/' + args.components.name,
                                      'bytes': args.components.stat().st_size, 'sha256': sha(args.components)}))
    index = {'repository': 'faitholopade/p3-labs-techarena-2026',
             'purpose': 'Research backup, not a competition submission.',
             'included': [record for _, record in included], 'repository_backed': external,
             'other_prerequisites': ['Private V11-V20 historical runtime archives',
                                     'V25-V28 research checkpoint archive', 'V29 research checkpoint archive',
                                     'Repository V27 and V29 full heads, V31 specification and source'],
             'omissions': 'Derived large raw feature matrices and duplicate evaluation outputs are reproducible and excluded.'}
    temporary = args.out.with_suffix('.zip.partial')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, record in included:
            archive.write(path, 'v30_v31/' + record['path'])
        archive.writestr('v30_v31/checkpoint_index.json', json.dumps(index, indent=2) + '\n')
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(included) + 1
        for _, record in included:
            assert hashlib.sha256(archive.read('v30_v31/' + record['path'])).hexdigest() == record['sha256']
    os.replace(temporary, args.out)
    report = {'filename': args.out.name, 'bytes': args.out.stat().st_size, 'sha256': sha(args.out),
              'files': len(included) + 1, 'zip_crc_valid': True, 'member_hashes_verified': True,
              'excluded_repo_backed_files': len(external), 'prefix': 'v30_v31',
              'purpose': 'All completed V30 fold checkpoints, predictions and V31 evidence. Not a competition submission.'}
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
