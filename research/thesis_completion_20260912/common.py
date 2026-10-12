from pathlib import Path
import hashlib
import json
import os

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = Path(os.environ.get('PM25_ORIGINAL_ROOT', r'C:\Users\USER\.codex\.chatgpt-projects\g-p-69aaacd3007481919301f520a145f3f4'))
MONTHLY = HERE.parent / 'common_monthly_transfer_20260912'
DATA = HERE / 'data'
AUDIT = HERE / 'audit'
RESULTS = HERE / 'results'
for p in (DATA, AUDIT, RESULTS):
    p.mkdir(parents=True, exist_ok=True)

def dump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()
