"""Diagnose day completeness from raw hourly records, not a changed threshold."""
import calendar
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=Path.home()/'.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'
RAW=ROOT/'data/raw/cnemc_zenodo_10836299'
for name in ['audit','data','evidence']:(HERE/name).mkdir(exist_ok=True)

def sha(p):
    with open(p,'rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def parse_file(frame,date):
    if not {'date','hour','type'}.issubset(frame.columns):
        rows=pd.DataFrame(columns=['station_code','valid_hours','daily_pm25','date','valid_day'])
        return rows,{'date':date,'parse_status':'invalid_header','pm25_rows':0,'distinct_pm25_hours':0,
            'missing_hour_numbers':'|'.join(map(str,range(24))),'stations_with_pm25_columns':0,
            'stations_reaching_20h':0,'maximum_station_valid_hours':0,'station_missing_cells':0,
            'station_out_of_range_cells':0,'header_preview':str(list(frame.columns))[:250]}
    q=frame[frame.type.astype(str).str.strip().eq('PM2.5')].copy()
    hours=pd.to_numeric(q.hour,errors='coerce')
    invalid=~hours.between(0,23)|hours.mod(1).ne(0)
    assert not invalid.any(),(date,'invalid PM2.5 hour')
    assert q.date.astype(str).eq(date).all(),(date,'date mismatch')
    assert not hours.duplicated().any(),(date,'duplicated PM2.5 hour')
    cols=[c for c in q.columns if re.fullmatch(r'\d+[A-Za-z]',c)]
    value=q[cols].apply(pd.to_numeric,errors='coerce')
    valid=value.ge(0)&value.le(1000)&np.isfinite(value)
    count=valid.sum(axis=0);means=value.where(valid).mean(axis=0)
    rows=pd.DataFrame({'station_code':cols,'valid_hours':count.to_numpy(),'daily_pm25':means.to_numpy()})
    rows['date']=date;rows['valid_day']=rows.valid_hours.ge(20)
    diag={'date':date,'parse_status':'valid_header','pm25_rows':len(q),'distinct_pm25_hours':len(hours.unique()),
          'missing_hour_numbers':'|'.join(map(str,sorted(set(range(24))-set(hours.astype(int))))),
          'stations_with_pm25_columns':len(cols),'stations_reaching_20h':int(count.ge(20).sum()),
          'maximum_station_valid_hours':int(count.max()) if len(count) else 0,
          'station_missing_cells':int(value.isna().sum().sum()),
          'station_out_of_range_cells':int((value.notna()&~valid).sum().sum())}
    return rows,diag

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    path=RAW/'20180101-20181231_CNEMC.zip'
    with open(path,'rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
    assert md5=='5fe71b95400f25922037421d16805ce6'
    daily=[];audit=[]
    with zipfile.ZipFile(path) as z:
        members={re.search(r'china_sites_(\d{8})\.csv$',n).group(1):n for n in z.namelist() if re.search(r'china_sites_(\d{8})\.csv$',n)}
        expected=set(pd.date_range('2018-01-01','2018-12-31').strftime('%Y%m%d'))
        assert len(members)==365 and set(members)==expected
        for date,member in sorted(members.items()):
            if date[4:6] not in ['08','12']:continue
            with z.open(member) as f:frame=pd.read_csv(f,low_memory=False)
            rows,diag=parse_file(frame,date)
            payload=z.read(member)
            diag['file_bytes']=len(payload);diag['file_sha256']=hashlib.sha256(payload).hexdigest()
            diag['html_404_payload']=payload.lstrip().startswith(b'<html>') and b'404 Not Found' in payload
            if diag['parse_status']=='invalid_header':
                (HERE/'evidence'/f'archive_china_sites_{date}.txt').write_bytes(payload)
            if not rows.empty:daily.append(rows)
            audit.append(diag)
    d=pd.concat(daily,ignore_index=True)
    d.to_csv(HERE/'data/audited_2018_aug_dec_station_days.csv.gz',index=False)
    a=pd.DataFrame(audit);a.to_csv(HERE/'audit/raw_day_availability.csv',index=False)
    d['month']=d.date.str[4:6].astype(int)
    ok=d[d.valid_day]
    monthly=ok.groupby(['station_code','month']).agg(recomputed_pm25=('daily_pm25','mean'),recomputed_days=('valid_day','size'),recomputed_hours=('valid_hours','sum')).reset_index()
    cached_path=ROOT/'model_outputs/national_cnemc_validation_20260824/tables/national_cnemc_station_monthly_2018.csv.gz'
    cached=pd.read_csv(cached_path);cached=cached[cached.month.isin([8,12])]
    compare=cached.merge(monthly,on=['station_code','month'],how='left',validate='one_to_one')
    assert compare.recomputed_days.notna().all()
    assert compare.valid_days.eq(compare.recomputed_days).all()
    assert compare.valid_hours.eq(compare.recomputed_hours).all()
    error=float((compare.pm25_mean_ug_m3-compare.recomputed_pm25).abs().max())
    assert error<1e-10,error
    compare.to_csv(HERE/'audit/raw_vs_cached_station_months.csv.gz',index=False)
    a['month']=a.date.str[4:6].astype(int)
    summary=[]
    for m,g in a.groupby('month'):
        summary.append({'month':int(m),'daily_files':len(g),'days_with_20_pm25_hour_rows':int(g.distinct_pm25_hours.ge(20).sum()),
                        'days_with_any_20h_station':int(g.stations_reaching_20h.gt(0).sum()),
                        'pm25_hour_rows':int(g.pm25_rows.sum()),'required_valid_days':24,
                        'maximum_cached_station_valid_days':int(compare[compare.month.eq(m)].valid_days.max()),
                        'global_hour_row_deficit_days':int(g.distinct_pm25_hours.lt(20).sum()),
                        'invalid_header_files':int(g.parse_status.eq('invalid_header').sum()),
                        'html_404_files':int(g.html_404_payload.sum())})
    pd.DataFrame(summary).to_csv(HERE/'audit/raw_month_diagnosis.csv',index=False)
    dump(HERE/'audit/raw_audit_receipt.json',{'archive':str(path),'archive_sha256':sha(path),'archive_md5_verified':True,
        'all_365_daily_members_present':True,'inspected_days':62,'cached_monthly_file':str(cached_path),
        'cached_sha256':sha(cached_path),'compared_station_months':len(compare),'maximum_mean_error':error,
        'daily_duplicate_hours':0,'invalid_hour_codes':0,'summary':summary,'original_files_modified':False})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Monthly reproduction maximum error',error,flush=True)

if __name__=='__main__':main()
