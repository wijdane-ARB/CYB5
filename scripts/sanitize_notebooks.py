#!/usr/bin/env python3
"""Check or clear committed Jupyter execution outputs without changing code cells.

Use --check to list notebooks containing stored output. --clean creates a .bak
copy (ignored or stored outside Git) before clearing output and execution_count.
Does not erase committed Git history: audit it separately if sensitive.
"""
import argparse
import json
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--clean', action='store_true')
    ap.add_argument('paths', nargs='+', type=Path)
    args = ap.parse_args()
    found = 0
    for path in args.paths:
        if path.suffix.lower() != '.ipynb':
            raise ValueError(f'Expected .ipynb: {path}')
        notebook = json.loads(path.read_text(encoding='utf-8'))
        cells = notebook.get('cells', [])
        dirty = [c for c in cells if c.get('cell_type') == 'code'
                 and (c.get('outputs') or c.get('execution_count') is not None)]
        print(f'{path}: {len(dirty)} executed/output-bearing cells')
        found += len(dirty)
        if args.clean and dirty:
            backup = path.with_suffix('.ipynb.bak')
            if backup.exists():
                raise FileExistsError(f'Backup already exists, refusing to overwrite: {backup}')
            shutil.copy2(path, backup)
            for cell in cells:
                if cell.get('cell_type') == 'code':
                    cell['outputs'] = []
                    cell['execution_count'] = None
            path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
            print('  Cleaned; backup saved at:', backup)
    if args.check and found:
        print('NOTE: inspect stored outputs for private data before committing; --clean removes all.')

if __name__ == '__main__':
    main()
