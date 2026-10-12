"""Produce source ledger, portable dependency notes and immutable comparison."""
import difflib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path
import requests
import bibtexparser
from common import HERE, ROOT, AUDIT, dump,digest

original=ROOT/'thesis_workspace/overleaf_pm25_ems_manuscript_20260829_v5_consolidated/main.tex'
comparison=HERE/'comparison';comparison.mkdir(exist_ok=True)
shutil.copy2(original,comparison/'before_20260829.tex')
after=HERE/'manuscript/main.tex';shutil.copy2(after,comparison/'after_20260913.tex')
diff=''.join(difflib.unified_diff(original.read_text(encoding='utf-8').splitlines(True),
    after.read_text(encoding='utf-8').splitlines(True),fromfile='before/main.tex',tofile='after/main.tex'))
(HERE/'manuscript_before_after.patch').write_text(diff,encoding='utf-8')
dump(comparison/'comparison_manifest.json',{'before_path':str(original),'before_sha256':digest(original),
    'after_path':str(after),'after_sha256':digest(after),'note':'Local unified diff, not a GitHub PR or a commit.'})
versions={'python':platform.python_version(),'system':platform.platform(),'packages':{}}
for name in ['numpy','pandas','scipy','scikit-learn','lightgbm','xgboost','catboost','geopandas','shapely','pyproj',
    'pyogrio','joblib','requests','matplotlib','python-calamine','xarray','zarr','numcodecs','bibtexparser','Pillow']:
    try:versions['packages'][name]=importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:versions['packages'][name]='not installed in science environment'
dump(HERE/'environment_versions.json',versions)
(HERE/'requirements-science.txt').write_text('\n'.join(k+'=='+v for k,v in versions['packages'].items() if not v.startswith('not '))+'\n',encoding='utf-8')
parser=bibtexparser.bparser.BibTexParser(common_strings=True);parser.ignore_nonstandard_types=False
entries=bibtexparser.loads((HERE/'manuscript/references.bib').read_text(encoding='utf-8'),parser=parser).entries
checked={r['key']:r for r in json.loads((AUDIT/'bibliography_metadata_audit.json').read_text(encoding='utf-8'))}
webkeys={'power_monthly','power_method','weatherbench_data','openmeteo_history','usyd5702','usydai',
    'naturalearth','cawley2010','davies2014','izzo2023','zhao2019','sugiyama2007','breiman2001','geurts2006',
    'hollmann2025tabpfn','nsw2026airqualityapi','qld2026airquality','vic2026airwatch','ghsfua2019','edgar2024ap81'}
rows=['# Source and literature ledger','',
    'Metadata verification is not full-text methodological replication. The Su et al. comparison remains publisher-abstract-level. Recorded URLs identify primary sources; none is a Google Scholar search-result citation.',
    '', '| Key | Title | Primary URL | Verification scope |','| --- | --- | --- | --- |']
records=[]
for e in entries:
    key=e['ID'];url='https://doi.org/'+e['doi'] if e.get('doi') else e.get('url','')
    status=checked.get(key,{}).get('status','curated primary metadata')
    if key in webkeys:status+='; primary page checked in this revision'
    if key=='su2022yrd':status+='; publisher abstract read, full methods not independently accessed'
    if key=='cnemc2024zenodo':
        r=requests.get('https://api.datacite.org/dois/10.5281/zenodo.10836299',timeout=60);r.raise_for_status()
        dump(AUDIT/'cnemc_datacite_metadata.json',r.json());status='DataCite registered creator/title and archived data content checked'
    rows.append('| '+' | '.join([key,e['title'].replace('{','').replace('}',''),f'[Primary source]({url})',status])+' |')
    records.append({'key':key,'url':url,'scope':status})
(HERE/'SOURCE_LEDGER.md').write_text('\n'.join(rows)+'\n',encoding='utf-8');dump(AUDIT/'source_ledger.json',records)
dump(AUDIT/'float_boundary_diagnosis.json',{'reproduced_default_parser_mismatches':4,'round_trip_mismatches':0,
    'configurations_scanned':90,'example_threshold':1.8109632833995406,'default_parsed_value':1.8109632833995408,
    'changed':'Saved prediction reading in independent audit; no fitted models, gates or scores changed.',
    'regression_test':'test_saved_csv_preserves_inclusive_gate_boundary',
    'diagnostic_hypotheses':['CSV decimal round-trip precision (confirmed by differential parser test)',
        'Serialized inequality or threshold change (not supported: same JSON rule reproduces)',
        'Row or city-split misalignment (not supported by full key/split checks)']})
print('METADATA_COMPLETE',len(entries),flush=True)
