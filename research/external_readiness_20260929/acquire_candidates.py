"""Download potential new cohorts; do not display or evaluate their outcomes."""
import csv
import io
import json
import re
import sys
import time
import zipfile
import requests
from common import HERE, digest, dump, now

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    search=json.loads((HERE/'evidence/sa_air_search.json').read_text(encoding='utf-8'))
    folder=HERE/'data/candidate_raw';folder.mkdir(exist_ok=True)
    inventories=[];downloads=[]
    ledger_path=HERE/'audit/candidate_downloads.json'
    old=json.loads(ledger_path.read_text(encoding='utf-8')) if ledger_path.exists() else []
    for package in search['result']['results']:
        if 'particle' not in package['name']:continue
        for resource in package['resources']:
            row={'provider':'SA_EPA','dataset':package['name'],'title':resource['name'],
                'url':resource['url'],'license':package.get('license_id'),
                'resource_id':resource['id'],'catalogue_modified':resource.get('last_modified')}
            inventories.append(row)
            if not re.search(r'202[34]',resource['name']):continue
            path=folder/(resource['id']+'.zip')
            item=dict(row,retrieved_at=now())
            cached=next((x for x in old if x.get('resource_id')==resource['id'] and x['status']=='downloaded'),None)
            if cached and path.exists():downloads.append(cached);continue
            try:
                r=requests.get(resource['url'],timeout=60);r.raise_for_status()
                if not zipfile.is_zipfile(io.BytesIO(r.content)):raise ValueError('Expected ZIP, not a validated archive')
                path.write_bytes(r.content)
                with zipfile.ZipFile(path) as z:
                    members=[]
                    for member in z.namelist():
                        if member.endswith('/'):continue
                        # Save schema lines only. No concentrations or score summaries are emitted.
                        with z.open(member) as stream:
                            lines=[]
                            for _ in range(6):
                                line=stream.readline().decode('utf-8-sig',errors='replace').strip()
                                if not line:break
                                if re.match(r'^"?\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',line):break
                                lines.append(line)
                        members.append({'name':member,'bytes':z.getinfo(member).file_size,'schema_lines':lines})
                item.update(status='downloaded',bytes=path.stat().st_size,sha256=digest(path),path=str(path.relative_to(HERE)),members=members,
                    outcome_use='schema_only_no_predictive_evaluation')
            except Exception as error:item.update(status='failed',error=str(error))
            downloads.append(item);dump(ledger_path,downloads)
            print('SA',resource['name'],item['status'],flush=True)
            time.sleep(.3)
    dump(HERE/'audit/sa_resource_inventory.json',inventories)
    dump(ledger_path,downloads)
    # Three fixed year-end availability probes, not a claim of complete China coverage.
    probes=[]
    for date in ('20231231','20241231','20251231'):
        url=f'https://quotsoft.net/air/data/china_sites_{date}.csv'
        path=folder/f'china_sites_{date}.csv'
        item={'provider':'quotsoft_secondary_CNEMC_archive','date':date,'url':url,'retrieved_at':now()}
        try:
            if not path.exists():
                r=requests.get(url,timeout=60);r.raise_for_status();path.write_bytes(r.content)
            with path.open(encoding='utf-8-sig',newline='') as stream:
                reader=csv.reader(stream);header=next(reader)
                if header[:3]!=['date','hour','type']:raise ValueError('Unexpected CSV schema')
                stamps=[]
                for row in reader:
                    if row[2]=='PM2.5':stamps.append((row[0],row[1]))
            assert all(x[0]==date for x in stamps)
            item.update(status='downloaded',bytes=path.stat().st_size,sha256=digest(path),
                path=str(path.relative_to(HERE)),station_columns=len(header)-3,pm25_hour_rows=len(stamps),
                unique_pm25_hours=len(set(stamps)),outcome_use='date_hour_type_only_no_concentration_analysis')
        except Exception as error:item.update(status='failed',error=str(error))
        probes.append(item);dump(HERE/'audit/china_availability_probes.json',probes)
        print('China probe',date,item['status'],flush=True)

if __name__=='__main__':main()
