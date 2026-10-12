import json
import pandas as pd
import zipfile
from common import ROOT, AUDIT, dump

vic = ROOT / 'data/raw/australia_ground_monitoring/vic/hourly_airwatch'
report = {}
for year in (2015, 2018, 2020, 2022):
    p = vic / f'{year}_All_sites_air_quality_hourly_avg.xlsx'
    x = pd.ExcelFile(p)
    sheets = x.sheet_names
    info = {'sheets': sheets}
    for s in sheets:
        if s in ('Data Info Panel', 'Metadata'):
            info[s] = pd.read_excel(x, sheet_name=s, header=None, nrows=40).fillna('').astype(str).values.tolist()
    first = next(s for s in sheets if s not in ('Data Info Panel', 'Metadata', 'AllData'))
    info['sample'] = pd.read_excel(x, sheet_name=first, nrows=3).to_dict(orient='records')
    report[str(year)] = info
    print(year, json.dumps(info, default=str, ensure_ascii=False)[:12000], flush=True)
dump(AUDIT/'vic_schema_inspection.json', report)
