"""Append missing year-end records in a separate branch, preserving frozen inputs."""
import calendar
import gzip
import json
import math
import time
import urllib.request
import numpy as np
import pandas as pd
from common import HERE, ROOT, BASELINE, digest, dump, now

API = 'https://data.airquality.nsw.gov.au/api/Data/get_Observations'
RAW = ROOT/'data/raw/australia_ground_monitoring/nsw/daily_pm25'
PATCH = HERE/'data/nsw_boundary_raw'
PATCH.mkdir(exist_ok=True)

def load_corrected(year):
    with gzip.open(RAW/f'nsw_pm25_daily_{year}.json.gz','rt',encoding='utf-8') as f:
        old=pd.DataFrame(json.load(f))
    response=json.loads((PATCH/f'nsw_year_end_{year}.json').read_text(encoding='utf-8'))
    new=pd.DataFrame(response)
    keys=['Site_Id','Date']
    assert not old.duplicated(keys).any() and not new.duplicated(keys).any()
    added=new.merge(old[keys].assign(already_present=True),on=keys,how='left')
    added=added[added.already_present.isna()].drop(columns='already_present')
    added=added[added.Date.eq(f'{year}-12-31')]
    result=pd.concat([old,added],ignore_index=True)
    assert not result.duplicated(keys).any()
    return old,result,added

def main():
    audit=[];changes=[];sources=[];overlaps=[]
    for year in range(2015,2026):
        original=RAW/f'nsw_pm25_daily_{year}.json.gz'
        with gzip.open(original,'rt',encoding='utf-8') as f: old=pd.DataFrame(json.load(f))
        sites=sorted(map(int,old.Site_Id.unique()))
        body={'Parameters':['PM2.5'],'Sites':sites,'StartDate':f'{year}-12-30',
              'EndDate':f'{year+1}-01-01','Categories':['Averages'],'SubCategories':['Daily'],
              'Frequency':['24h average derived from 1h average']}
        path=PATCH/f'nsw_year_end_{year}.json'
        request_path=PATCH/f'nsw_year_end_{year}_request.json'
        if not path.exists():
            for attempt in range(3):
                try:
                    req=urllib.request.Request(API,json.dumps(body).encode(),{'Content-Type':'application/json','User-Agent':'PM25-dissertation-data-audit/1.0'})
                    with urllib.request.urlopen(req,timeout=90) as response:
                        payload=json.loads(response.read())
                    if isinstance(payload,dict):payload=payload['Values']
                    assert isinstance(payload,list) and payload
                    dump(path,payload)
                    dump(request_path,{'url':API,'retrieved_at':now(),'body':body,'response_sha256':digest(path)})
                    break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2**attempt)
        meta=json.loads(request_path.read_text())
        assert meta['response_sha256']==digest(path) and meta['body']==body
        old,corrected,added=load_corrected(year)
        response=pd.DataFrame(json.loads(path.read_text()))
        assert set(response.Date)=={f'{year}-12-30',f'{year}-12-31'}
        assert set(added.Site_Id)==set(sites) and len(added)==len(sites)
        assert response.Parameter.map(lambda x:x['ParameterCode']=='PM2.5' and x['SubCategory']=='Daily' and x['Frequency']=='24h average derived from 1h average').all()
        overlap=old.merge(response,on=['Site_Id','Date'],suffixes=('_old','_new'))
        equal=np.isclose(pd.to_numeric(overlap.Value_old),pd.to_numeric(overlap.Value_new),equal_nan=True)
        overlap['unchanged']=equal
        overlap['year']=year
        overlaps.append(overlap[['year','Site_Id','Date','Value_old','Value_new','unchanged']])
        expected=set(pd.date_range(f'{year}-01-01',f'{year}-12-31').strftime('%Y-%m-%d'))
        assert set(corrected.Date)==expected
        counts=corrected.groupby('Site_Id').Date.nunique()
        assert counts.eq(len(expected)).all()
        valid=pd.to_numeric(added.Value,errors='coerce').between(-10,1000)
        audit.append({'year':year,'original_calendar_dates':old.Date.nunique(),'expected_dates':len(expected),
                      'added_station_day_slots':len(added),'added_valid_daily_pm25':int(valid.sum()),
                      'corrected_calendar_dates':corrected.Date.nunique(),'overlap_changed_values':int((~equal).sum()),
                      'missing_date_before':f'{year}-12-31','missing_dates_after':0})
        for site in sites:
            values=[]
            for frame in (old,corrected):
                v=pd.to_numeric(frame.loc[frame.Site_Id.eq(site)&frame.Date.str.startswith(f'{year}-12'),'Value'],errors='coerce')
                v=v[v.between(-10,1000)]
                values.append((float(v.mean()) if len(v) else np.nan,len(v)))
            (a,na),(b,nb)=values
            changes.append({'site_key':f'NSW_{site}','year':year,'month':12,'before_pm25':a,'after_pm25':b,
                            'before_valid_days':na,'after_valid_days':nb,'before_eligible':na>=24,'after_eligible':nb>=24,
                            'delta_pm25':b-a,'new_valid_day':nb-na})
        sources.append({'original_path':str(original),'original_sha256':digest(original),'patch_path':str(path),
                        'patch_sha256':digest(path),'request_path':str(request_path)})
        print(year,'added',len(added),'valid',int(valid.sum()),'overlap changed',int((~equal).sum()),flush=True)
    a=pd.DataFrame(audit);c=pd.DataFrame(changes)
    a.to_csv(HERE/'audit/nsw_calendar_boundary_audit.csv',index=False)
    c.to_csv(HERE/'audit/nsw_december_before_after.csv',index=False)
    pd.concat(overlaps).to_csv(HERE/'audit/nsw_overlap_revision_check.csv',index=False)
    baseline_path=BASELINE/'data/regulatory_station_month_2015_2022.csv.gz'
    baseline=pd.read_csv(baseline_path)
    old=baseline[baseline.site_key.str.startswith('NSW_')&baseline.month.eq(12)]
    v=old.merge(c[c.year.le(2022)],on=['site_key','year','month'],validate='one_to_one')
    assert len(v)==len(old)
    assert np.allclose(v.observed_pm25,v.before_pm25,equal_nan=True)
    v.to_csv(HERE/'audit/frozen_baseline_december_impact.csv',index=False)
    # This is a correction candidate, never a replacement for the original study input.
    corrected=baseline.merge(c[c.year.le(2022)][['site_key','year','month','after_pm25','after_valid_days']],
                             on=['site_key','year','month'],how='left',validate='one_to_one')
    mask=corrected.after_valid_days.notna()
    corrected.loc[mask,'observed_pm25']=corrected.loc[mask,'after_pm25']
    corrected.loc[mask,'valid_days']=corrected.loc[mask,'after_valid_days']
    corrected['eligible']=corrected.valid_days>=corrected.required_days
    corrected=corrected.drop(columns=['after_pm25','after_valid_days'])
    corrected.to_csv(HERE/'data/regulatory_station_month_existing_rows_boundary_corrected_2015_2022.csv.gz',index=False)
    dump(HERE/'audit/nsw_boundary_repair_summary.json',{'created_at':now(),'source_files':sources,
        'baseline_sha256':digest(baseline_path),'added_station_day_slots':int(a.added_station_day_slots.sum()),
        'added_valid_daily_pm25':int(a.added_valid_daily_pm25.sum()),'changed_december_means':int(c.delta_pm25.abs().gt(1e-12).sum()),
        'newly_eligible_december_months':int((~c.before_eligible&c.after_eligible).sum()),
        'overlap_revision_count':int(a.overlap_changed_values.sum()),
        'original_inputs_modified':False,'old_model_results_recomputed':False,
        'corrected_old_table_scope':'existing rows only; old filtered-out months are not reinstated here; rebuild full old cohort before modelling'})

if __name__=='__main__':main()
