from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
ROOT = Path.home()/'.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'
PREVIOUS = HERE.parent/'regional_station_scope_20260920'
BASELINE = HERE.parent/'thesis_completion_20260912'
OUT = WORK/'output/research/external_readiness_20260929'
for directory in (HERE/'evidence', HERE/'data', HERE/'audit', OUT):
    directory.mkdir(parents=True, exist_ok=True)

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()

def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding='utf-8')

def now():
    return datetime.now(timezone.utc).isoformat()
