"""Independent aggregation checks and availability-only inspection of new archives."""
import calendar
import csv
import gzip
import io
import json
import re
import sys
import zipfile
import numpy as np
import pandas as pd
from common import HERE, ROOT, PREVIOUS, BASELINE, digest, dump, now

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    manifest=json.loads((HERE/'audit/extension_manifest.json').read_text(encoding='utf-8'))
    for source in manifest['source_files']:
        assert digest(source['path'])==source['sha256'],source['path']
    evidence=json.loads((HERE/'audit/exposure_evidence.json').read_text(encoding='utf-8'))
    for source in evidence['files']:
        assert digest(source['path'])==source['sha256'],source['path']
    assert digest(HERE/'PROTOCOL.md')==manifest['protocol_sha256']
    assert digest(HERE/'build_extension.py')==manifest['script_sha256']
    monthly=pd.read_csv(HERE/'data/australia_station_month_2023_2025.csv.gz')
    assert not monthly.duplicated(['site_key','year','month']).any()
    assert set(monthly.year)=={2023,2024,2025}
    assert set(monthly.network_state)=={'NSW','QLD','VIC'}
    assert monthly.observed_pm25.between(-10,1000).all()
    assert monthly.longitude.notna().all() and monthly.latitude.notna().all()
    for r in monthly.itertuples():
        assert r.required_days==int(np.ceil(.75*calendar.monthrange(r.year,r.month)[1]))
        assert r.valid_days<=calendar.monthrange(r.year,r.month)[1]
        assert r.eligible==(r.valid_days>=r.required_days)
        if r.network_state!='NSW':assert r.valid_days*20<=r.valid_hours<=r.valid_days*24
    f=pd.read_csv(HERE/'data/australia_fua_month_2023_2025.csv.gz')
    raw=monthly[monthly.eligible&monthly.eFUA_ID.notna()]
    keys=['country_iso','eFUA_ID','year','month']
    independent=raw.groupby(keys).observed_pm25.apply(lambda x:sum(x)/len(x)).rename('recomputed').reset_index()
    compare=f.merge(independent,on=keys,validate='one_to_one')
    fua_max=float((compare.observed_pm25-compare.recomputed).abs().max())
    assert len(compare)==len(f) and fua_max<1e-10
    qa=[]
    au=ROOT/'data/raw/australia_ground_monitoring'
    # A separate raw-data calculation, not a call to the production aggregate function.
    def check(site,year,month,dates,values,hourly=True):
        dates=pd.to_datetime(dates,errors='coerce')
        values=pd.to_numeric(values,errors='coerce')
        t=pd.DataFrame({'date':dates,'value':values})
        t=t[t.date.dt.year.eq(year)&t.date.dt.month.eq(month)&t.value.between(-10,1000)]
        if hourly:
            t=t.drop_duplicates('date');t['day']=t.date.dt.floor('D')
            daily=[]
            for day,g in t.groupby('day'):
                if len(g)>=20:daily.append((float(np.mean(g.value)),len(g)))
            value=float(np.mean([x[0] for x in daily]));days=len(daily);hours=sum(x[1] for x in daily)
        else:value=float(np.mean(t.value));days=len(t);hours=None
        result=monthly[monthly.site_key.eq(site)&monthly.year.eq(year)&monthly.month.eq(month)].iloc[0]
        error=abs(result.observed_pm25-value)
        assert error<1e-10 and result.valid_days==days
        if hourly:assert result.valid_hours==hours
        qa.append({'site_key':site,'year':year,'month':month,'absolute_recalculation_error':error,'valid_days_match':True,'hour_count_match':True if hourly else 'unavailable_NSW'})
    for year in (2023,2024,2025):
        with gzip.open(au/f'nsw/daily_pm25/nsw_pm25_daily_{year}.json.gz','rt',encoding='utf-8') as stream:
            nsw=pd.DataFrame(json.load(stream))
        patch_path=HERE/f'data/nsw_boundary_raw/nsw_year_end_{year}.json'
        patch=pd.DataFrame(json.loads(patch_path.read_text(encoding='utf-8')))
        patch=patch[patch.Date.eq(f'{year}-12-31')]
        nsw=pd.concat([nsw,patch],ignore_index=True)
        assert not nsw.duplicated(['Site_Id','Date']).any()
        expected=366 if calendar.isleap(year) else 365
        assert nsw.groupby('Site_Id').Date.nunique().eq(expected).all()
        nsw=nsw[nsw.Site_Id.eq(107)]
        check('NSW_107',year,1,nsw.Date,nsw.Value,False)
        check('NSW_107',year,12,nsw.Date,nsw.Value,False)
        qld=pd.read_csv(au/f'qld/hourly_pm25/pm2-5-qld-{year}.csv',low_memory=False)
        col=next(x for x in qld if x.startswith('Rocklea ('))
        dates=pd.to_datetime(qld.Date.astype(str)+' '+qld.Time.astype(str),dayfirst=True)
        check('QLD_rocklea',year,1,dates,qld[col])
        x=pd.ExcelFile(au/f'vic/hourly_airwatch/{year}_All_sites_air_quality_hourly_avg.xlsx',engine='calamine')
        if 'Metadata' in x.sheet_names:
            s=next(x for x in x.sheet_names if x.startswith('Alphington_'))
            v=pd.read_excel(x,sheet_name=s);site='VIC_'+str(int(v.location_id.dropna().iloc[0]))
            dt='datetime_local' if 'datetime_local' in v else 'datetime_AEST'
            check(site,year,1,v[dt],v['SBPM25'])
        else:
            s=next(s for s in x.sheet_names if s!='Data Info Panel')
            v=pd.read_excel(x,sheet_name=s)
            v=v[v.sp_name.str.lower().eq('alphington')&v.param_id.eq('SBPM25')&v.time_basis_id.eq('1HR_AV')]
            site='VIC_'+str(int(v.sample_point_id.iloc[0]))
            check(site,year,1,v.sample_datetime,v.value)
    pd.DataFrame(qa).to_csv(HERE/'audit/independent_monthly_recalculation.csv',index=False)
    reg=pd.read_csv(HERE/'data/observed_station_year_registry_2015_2025.csv')
    assert not reg.duplicated(['site_key','year']).any()
    assert not reg[(reg.site_key=='VIC_10169')&reg.year.eq(2021)].role_usable_without_backfill.any()
    assert not reg[reg.country_iso.eq('CHN')].role_usable_without_backfill.any()
    assert not reg[reg.same_year_role_conflict].role_usable_without_backfill.any()
    annual=pd.read_csv(HERE/'data/qld_station_year_evidence_2015_2025.csv')
    assert len(annual)==496 and not annual.duplicated(['site_key','year']).any()
    assert annual[annual.site_key.eq('QLD_rangeville')].manufacturer_instruction_only.all()
    assert annual[annual.site_key.eq('QLD_wynnumnorth')&annual.year.eq(2022)].semivolatile_loss_uncompensated.all()
    boundary=pd.read_csv(HERE/'audit/nsw_calendar_boundary_audit.csv')
    assert len(boundary)==11 and boundary.missing_dates_after.eq(0).all()
    repair=json.loads((HERE/'audit/nsw_boundary_repair_summary.json').read_text())
    for source in repair['source_files']:
        assert digest(source['original_path'])==source['original_sha256']
        assert digest(source['patch_path'])==source['patch_sha256']
    corrected=pd.read_csv(HERE/'data/regulatory_station_month_existing_rows_boundary_corrected_2015_2022.csv.gz')
    assert corrected.eligible.eq(corrected.valid_days>=corrected.required_days).all()
    # The candidate archive is opened only for schema, timestamps and missingness.
    downloads=json.loads((HERE/'audit/candidate_downloads.json').read_text(encoding='utf-8'))
    availability=[]
    for item in downloads:
        if item['status']!='downloaded':continue
        path=HERE/item['path'];assert digest(path)==item['sha256']
        stamps=[];headers=set();slots=nonempty=0;members=0
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not member.lower().endswith('.csv'):continue
                with z.open(member) as stream:
                    reader=csv.reader(io.TextIOWrapper(stream,encoding='utf-8-sig'))
                    header=next(reader)
                    idx=[i for i,h in enumerate(header) if 'PM2.5' in h.upper()]
                    if not idx:continue
                    headers.update(header[i] for i in idx);members+=1
                    for row in reader:
                        if not row:continue
                        stamps.append(row[0]);slots+=1
                        nonempty+=any(i<len(row) and row[i].strip()!='' for i in idx)
        parsed=pd.to_datetime(pd.Series(stamps,dtype=str),dayfirst=True,errors='coerce')
        availability.append({'dataset':item['dataset'],'resource_title':item['title'],
            'pm25_csv_members':members,'pm25_timestamp_rows':slots,'nonempty_pm25_rows':nonempty,
            'invalid_timestamps':int(parsed.isna().sum()),'duplicate_timestamps':int(parsed.duplicated().sum()),
            'first_timestamp':parsed.min(),'last_timestamp':parsed.max(),
            'pm25_headers':'|'.join(sorted(headers)), 'sha256':item['sha256'],
            'readiness':'schema_only_not_QC_or_model_validated','source_url':item['url']})
    av=pd.DataFrame(availability)
    av.to_csv(HERE/'audit/sa_candidate_availability.csv',index=False)
    dump(HERE/'audit/VERIFICATION.json',{'verified_at':now(),'passed':True,
        'source_and_frozen_files_verified':len(manifest['source_files'])+len(evidence['files']),
        'station_month_rows':len(monthly),'fua_month_rows':len(f),'fua_aggregation_max_error':fua_max,
        'independent_raw_station_month_checks':len(qa),'max_raw_monthly_error':max(x['absolute_recalculation_error'] for x in qa),
        'historical_registry_rows':len(reg),'sa_archive_count':len(downloads),
        'sa_pm25_archives':int(av.pm25_csv_members.gt(0).sum()),
        'sa_pm25_timestamp_rows':int(av.pm25_timestamp_rows.sum()),
        'sa_nonempty_pm25_rows':int(av.nonempty_pm25_rows.sum()),
        'sa_predicted_or_scored':False,'model_selection_performed':False,
        'untouched_validation_certified':False})
    print('Verification passed;',len(qa),'raw months;',len(f),'FUA months;',len(av),'SA archives',flush=True)
    print(av[['resource_title','pm25_csv_members','pm25_timestamp_rows','first_timestamp','last_timestamp']].to_string(index=False),flush=True)

if __name__=='__main__':main()
