"""Check archived source identity and syntax without loading private artifacts."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'source_manifest.json').read_text())
    for entry in manifest['files']:
        path = ROOT / entry['path']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry['sha256']:
            raise ValueError('Archived file differs: ' + entry['path'])
    sources = []
    for folder in ('src', 'scripts', 'experiments', 'tests'):
        for path in (ROOT / folder).rglob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
            sources.append(path)
    for folder in ('artifacts', 'results', 'config', 'experiments'):
        for path in (ROOT / folder).rglob('*.json'):
            json.loads(path.read_text())
    print(f"Verified {len(manifest['files'])} archived files; parsed {len(sources)} Python files.")


if __name__ == '__main__':
    main()
