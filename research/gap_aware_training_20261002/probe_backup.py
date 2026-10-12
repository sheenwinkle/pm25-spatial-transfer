"""Deterministic public-archive availability probes; never merge silently."""
import hashlib
import io
import json
import re
import time
import zipfile
from datetime import datetime,timezone
import numpy as np
import pandas as pd
import requests
from audit_raw import HERE,RAW,parse_file,dump


def cells(frame):
    if not {'date','hour','type'}.issubset(frame.columns):return pd.Series(dtype=float)
    q=frame[frame.type.astype(str).str.strip().eq('PM2.5')].copy()
    c=[x for x in q.columns if re.fullmatch(r'\d+[A-Za-z]',x)]
    assert not q.duplicated(['date','hour']).any()
    z=q.set_index(['date','hour'])[c].apply(pd.to_numeric,errors='coerce')
    z=z.where(z.ge(0)&z.le(1000)&np.isfinite(z))
    return z.stack().dropna()


def main():
    audit=pd.read_csv(HERE/'audit/raw_day_availability.csv',dtype={'date':str})
    # First two hour-deficit days and first two 404 payloads; one complete control per month.
    bad=audit[(audit.distinct_pm25_hours.lt(20))&audit.parse_status.eq('valid_header')].head(2).date.tolist()
    bad+=audit[audit.html_404_payload].head(2).date.tolist()
    controls=[audit[(audit.date.str[4:6].eq(m))&audit.distinct_pm25_hours.eq(24)].iloc[0].date for m in ['08','12']]
    dates=bad+controls
    dump(HERE/'audit/probe_plan.json',{'dates':dates,'selected_without_model_results':True,
        'selection':'first two valid-header hour deficits, first two 404s, first complete control per month',
        'independent_measurement_source':False,'automatic_merge':False})
    root=HERE/'evidence/backup_probes';root.mkdir(exist_ok=True)
    session=requests.Session();session.headers['User-Agent']='Academic-data-integrity-audit/1.0'
    results=[]
    with zipfile.ZipFile(RAW/'20180101-20181231_CNEMC.zip') as archive:
        members={re.search(r'china_sites_(\d{8})\.csv$',n).group(1):n for n in archive.namelist() if re.search(r'china_sites_(\d{8})\.csv$',n)}
        for date in dates:
            path=root/f'{date}.body';receipt=root/f'{date}.json'
            if receipt.exists():
                row=json.loads(receipt.read_text(encoding='utf-8'));results.append(row);continue
            url=f'https://quotsoft.net/air/data/china_sites_{date}.csv'
            row={'date':date,'control':date in controls,'url':url,'retrieved_utc':datetime.now(timezone.utc).isoformat()}
            try:
                response=session.get(url,timeout=(15,45));payload=response.content;path.write_bytes(payload)
                row.update({'status_code':response.status_code,'final_url':response.url,'bytes':len(payload),
                    'sha256':hashlib.sha256(payload).hexdigest(),'content_type':response.headers.get('Content-Type','')})
                original=archive.read(members[date]);row['identical_bytes_to_original']=payload==original
                if response.ok and payload.startswith(b'date,hour,type,'):
                    new=pd.read_csv(io.BytesIO(payload),low_memory=False);old=pd.read_csv(io.BytesIO(original),low_memory=False)
                    _,diag=parse_file(new,date);a=cells(old);b=cells(new);both=a.index.intersection(b.index)
                    row.update({'valid_csv':True,'pm25_hour_rows':diag['pm25_rows'],
                        'stations_reaching_20h':diag['stations_reaching_20h'],'new_valid_station_hour_cells':len(b.index.difference(a.index)),
                        'overlap_cells':len(both),'conflicting_overlap_cells':int((a.loc[both]-b.loc[both]).abs().gt(1e-9).sum())})
                else:row.update({'valid_csv':False,'new_valid_station_hour_cells':0})
            except (requests.RequestException,ValueError,AssertionError) as exc:
                row['error']=str(exc);row['valid_csv']=False
            dump(receipt,row);results.append(row);print(date,row.get('status_code'),row.get('new_valid_station_hour_cells'),flush=True)
            time.sleep(1)
    pd.DataFrame(results).to_csv(HERE/'audit/backup_probe_results.csv',index=False)
    dump(HERE/'audit/backup_probe_receipt.json',{'probe_count':len(results),'valid_csvs':sum(r['valid_csv'] for r in results),
        'recovered_candidate_station_hours':sum(r.get('new_valid_station_hour_cells',0) for r in results),
        'automatic_merge':False,'exhaustive_source_search':False,'independent_measurement_source':False})


if __name__=='__main__':main()
