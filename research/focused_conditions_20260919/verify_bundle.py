"""Check the delivered zip from a separate extracted dependency tree."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from run_focused import HERE, sha, dump

WORK = HERE.parents[1]
release = WORK / 'output/research/focused_conditions_20260919'
bundle = release / 'focused_conditions_reproducibility.zip'
qa_root = (WORK / 'output/qa').resolve()
destination = (qa_root / ('conditions_bundle_' + sha(bundle)[:12])).resolve()
assert destination.is_relative_to(qa_root)
destination.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(bundle) as z:
    assert z.testzip() is None
    for name in z.namelist():
        target = (destination / name).resolve()
        assert target.is_relative_to(destination)
    z.extractall(destination)
new = destination / HERE.name
records = []
for args in [['-m', 'unittest', 'test_focused.py', '-v'], ['audit_focused.py'], ['audit_recency.py']]:
    r = subprocess.run([sys.executable] + args, cwd=new, text=True, encoding='utf-8', errors='replace', capture_output=True)
    log_name = 'test_focused.log' if args[0] == '-m' else args[0] + '.log'
    (destination / log_name).write_text(r.stdout + r.stderr, encoding='utf-8')
    records.append({'args': args, 'returncode': r.returncode, 'log': str(destination / log_name)})
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    print('BUNDLE_CHECK_OK', args, flush=True)
result = {'bundle_sha256': sha(bundle), 'extracted_to': str(destination), 'checks': records,
          'passed': True, 'checked_from_separate_tree': True}
dump(release / 'CLEAN_BUNDLE_VERIFICATION.json', result)
receipt_path = release / 'DELIVERY_RECEIPT.json'
receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
p = release / 'CLEAN_BUNDLE_VERIFICATION.json'
receipt[p.name] = {'bytes': p.stat().st_size, 'sha256': sha(p)}
dump(receipt_path, receipt)
print('BUNDLE_VERIFIED', sha(bundle))
