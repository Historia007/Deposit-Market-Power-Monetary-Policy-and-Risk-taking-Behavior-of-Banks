"""Validate this code-only archive without importing or executing research programs."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def require(ok, message):
    if not ok:
        raise AssertionError(message)

def main():
    programs = list((ROOT / 'code').rglob('*.py'))
    for path in programs:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    for row in json.loads((ROOT / 'docs/source_manifest.json').read_text()):
        actual = hashlib.sha256((ROOT / row['path']).read_bytes()).hexdigest()
        require(actual == row['packaged_sha256'], 'Source changed: ' + row['path'])
    docs = list(ROOT.rglob('*.md'))
    for path in docs:
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' in link or link.startswith('#'):
                continue
            require((path.parent / link.split('#')[0]).exists(), f'Broken link: {link} in {path.name}')
    allowed = {'.py', '.do', '.md', '.json', '.txt'}
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.git' in path.parts or '__pycache__' in path.parts:
            continue
        require(path.name == '.gitignore' or path.suffix in allowed, f'Unexpected upload file: {path}')
        if path.suffix in allowed:
            personal_prefix = '/' + 'Users/'
            require(personal_prefix not in path.read_text(), f'Personal path: {path}')
    print(f'PASS: {len(programs)} Python files parsed; source hashes, {len(docs)} Markdown files, relative links and code-only scope checked.')
    print('No data pipeline or regressions executed by this check.')

if __name__ == '__main__':
    main()
