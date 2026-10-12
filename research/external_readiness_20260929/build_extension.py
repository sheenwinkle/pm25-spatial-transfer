"""Quality-audited monthly extension; no modelling or score calculation."""
import calendar
import gzip
import json
import re
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from common import HERE, ROOT, BASELINE, PREVIOUS, digest, dump, now
from repair_nsw_boundary import load_corrected, PATCH

AU = ROOT/'data/raw/australia_ground_monitoring'
META = ['site_key','station_name','network_state','longitude','latitude']
FILES, QC, LOCATIONS, METHODS, UNMATCHED = [], [], [], [], []

def record(path):
    FILES.append({'path':str(path),'bytes':path.stat().st_size,'sha256':digest(path)})

def monthly(d):
    d=d.copy()
    d['year']=d.date.dt.year
    d['month']=d.date.dt.month
    z=d.groupby(META+['year','month'],dropna=False).agg(
        observed_pm25=('daily_mean','mean'),valid_days=('daily_mean','count'),
        valid_hours=('valid_hours','sum')).reset_index()
    z['required_days']=[int(np.ceil(.75*calendar.monthrange(int(y),int(m))[1])) for y,m in zip(z.year,z.month)]
    z['eligible']=z.valid_days>=z.required_days
    return z

def aggregate(h,tag,year):
    h=h.copy()
    h['value']=pd.to_numeric(h.value,errors='coerce')
    h['datetime']=pd.to_datetime(h.datetime,errors='coerce')
    finite=np.isfinite(h.value)
    QC.append({'source':tag,'raw_rows':len(h),'finite':int(finite.sum()),
        'missing_or_nonfinite':int((~finite).sum()),'negative':int((h.value<0).sum()),
        'out_of_range':int((finite&~h.value.between(-10,1000)).sum()),
        'invalid_time':int(h.datetime.isna().sum()),
        'out_of_calendar_year':int((h.datetime.notna()&h.datetime.dt.year.ne(year)).sum()),
        'duplicate_site_hours_before_method_priority':int(h.duplicated(['site_key','datetime']).sum())})
    h=h[finite&h.value.between(-10,1000)&h.datetime.notna()&h.datetime.dt.year.eq(year)].copy()
    h['rank']=h.method.map({'SBPM25':0,'BPM2.5':1,'PM2.5':2}).fillna(3)
    h=h.sort_values('rank',kind='stable').drop_duplicates(['site_key','datetime'])
    assert not h.duplicated(['site_key','datetime']).any()
    for (site,method),g in h.groupby(['site_key','method']):
        METHODS.append({'site_key':site,'year':year,'method':method,'retained_hours':len(g),
            'first_timestamp':g.datetime.min(),'last_timestamp':g.datetime.max()})
    h['date']=h.datetime.dt.floor('D')
    d=h.groupby(META+['date'],dropna=False).agg(daily_mean=('value','mean'),valid_hours=('value','size')).reset_index()
    return monthly(d[d.valid_hours>=20])

def qld(year):
    normal=lambda x:re.sub('[^a-z0-9]','',str(x).lower())
    mp=AU/'qld/qld_air_quality_station_year_metadata_2015_2024.csv'
    fp=AU/'qld/monitoring-sites-on-open-data.csv'
    record(mp);record(fp)
    meta=pd.read_csv(mp);meta['key']=meta.station_name.map(normal)
    ep=HERE/'data/qld_station_year_evidence_2015_2025.csv';record(ep)
    evidence=pd.read_csv(ep)
    fallback=pd.read_csv(fp)
    fallback=fallback[fallback['Monitoring type'].eq('Air quality')].copy()
    fallback['key']=fallback['Site name'].map(normal)
    path=AU/f'qld/hourly_pm25/pm2-5-qld-{year}.csv';record(path)
    raw=pd.read_csv(path,low_memory=False)
    dates=pd.to_datetime(raw.Date.astype(str)+' '+raw.Time.astype(str),dayfirst=True,errors='coerce')
    frames=[]
    for col in raw.columns[2:]:
        name=re.sub(r'\s*\(ug/m\^3\)\s*$','',col).split(' (')[0].strip()
        key=normal(name);m=meta[meta.key.eq(key)].copy();myear=None
        exact=evidence[evidence.site_key.eq('QLD_'+key)&evidence.year.eq(year)]
        if len(exact)==1 and exact[['longitude','latitude']].notna().all(axis=None):
            row=exact.iloc[0];lon,lat=row.longitude,row.latitude;myear=year
            status='same_year_official_resource_metadata'
        elif len(m):
            m['delta']=(m.year-year).abs()
            row=m.sort_values(['delta','year','resource_name']).iloc[0]
            lon,lat=row.station_longitude,row.station_latitude
            myear=int(row.year)
            status='same_year_catalogue' if myear==year else 'nearest_year_coordinate_assumption'
        else:
            m=fallback[fallback.key.eq(key)]
            if len(m):
                row=m.iloc[-1];lon,lat=row.Longitude,row.Latitude
                status='undated_station_directory_assumption'
            else:
                lon=lat=np.nan;status='missing_coordinate'
                UNMATCHED.append({'network_state':'QLD','year':year,'station_name':name})
        LOCATIONS.append({'site_key':'QLD_'+key,'year':year,'coordinate_status':status,'coordinate_metadata_year':myear})
        frames.append(pd.DataFrame({'datetime':dates,'value':raw[col],'method':'PM2.5',
            'site_key':'QLD_'+key,'station_name':name,'network_state':'QLD','longitude':lon,'latitude':lat}))
    d=aggregate(pd.concat(frames,ignore_index=True),f'QLD_{year}',year)
    d['daily_quality']='20h_recomputed'
    d['time_basis']='native_archive_day_timezone_unconfirmed'
    return d

def vic(year):
    path=AU/f'vic/hourly_airwatch/{year}_All_sites_air_quality_hourly_avg.xlsx';record(path)
    x=pd.ExcelFile(path,engine='calamine');frames=[]
    if 'Metadata' not in x.sheet_names:
        sheet=next(s for s in x.sheet_names if s!='Data Info Panel')
        raw=pd.read_excel(x,sheet_name=sheet)
        raw=raw[raw.time_basis_id.eq('1HR_AV')&raw.param_id.isin(['SBPM25','BPM2.5','PM2.5'])].copy()
        h=raw.rename(columns={'sample_datetime':'datetime','param_id':'method','sp_name':'station_name'})
        h['site_key']='VIC_'+h.sample_point_id.astype(int).astype(str);h['network_state']='VIC'
        frames.append(h[META+['datetime','method','value']])
        basis='legacy_native_calendar_day_timezone_unconfirmed'
    else:
        loc=pd.read_excel(x,sheet_name='Metadata',header=None)
        start=int(loc.index[loc.iloc[:,0].astype(str).eq('location_id')][0])
        loc=loc.iloc[start+1:,:4].copy();loc.columns=['id','name','lat','lon']
        loc['id']=pd.to_numeric(loc.id,errors='coerce');loc=loc.dropna(subset=['id']).set_index('id')
        assert not loc.index.duplicated().any()
        for sheet in x.sheet_names:
            if sheet in ['Metadata','Data Info Panel','AllData']:continue
            raw=pd.read_excel(x,sheet_name=sheet)
            methods=[c for c in ['SBPM25','BPM2.5','PM2.5'] if c in raw]
            if not methods or raw.empty:continue
            ids=raw.location_id.dropna().unique();assert len(ids)==1
            sid=int(ids[0]);row=loc.loc[sid]
            dtcol='datetime_local' if 'datetime_local' in raw else 'datetime_AEST'
            for method in methods:
                frames.append(pd.DataFrame({'datetime':raw[dtcol],'value':raw[method],'method':method,
                    'site_key':f'VIC_{sid}','station_name':row['name'],'network_state':'VIC',
                    'longitude':float(row.lon),'latitude':float(row.lat)}))
        basis='native_datetime_local_or_AEST_column'
    h=pd.concat(frames,ignore_index=True)
    for site in h.site_key.unique():
        LOCATIONS.append({'site_key':site,'year':year,'coordinate_status':'same_year_file_metadata','coordinate_metadata_year':year})
    d=aggregate(h,f'VIC_{year}',year)
    d['daily_quality']='20h_recomputed';d['time_basis']=basis
    return d

def nsw(year):
    mp=AU/'nsw/nsw_air_quality_site_details.json';record(mp)
    meta=pd.DataFrame(json.loads(mp.read_text(encoding='utf-8')))
    path=AU/f'nsw/daily_pm25/nsw_pm25_daily_{year}.json.gz';record(path)
    record(PATCH/f'nsw_year_end_{year}.json')
    _,raw,_=load_corrected(year)
    expected=366 if calendar.isleap(year) else 365
    assert raw.groupby('Site_Id').Date.nunique().eq(expected).all()
    raw=raw.merge(meta[['Site_Id','SiteName','Longitude','Latitude']],on='Site_Id',validate='many_to_one',how='left')
    d=raw.rename(columns={'SiteName':'station_name','Longitude':'longitude','Latitude':'latitude','Value':'daily_mean'})
    d['daily_mean']=pd.to_numeric(d.daily_mean,errors='coerce');d['date']=pd.to_datetime(d.Date)
    d['site_key']='NSW_'+d.Site_Id.astype(str);d['network_state']='NSW';d['valid_hours']=np.nan
    assert not d.duplicated(['site_key','date']).any()
    QC.append({'source':f'NSW_{year}','raw_daily_rows':len(d),'finite':int(np.isfinite(d.daily_mean).sum()),
        'negative':int((d.daily_mean<0).sum()),'out_of_range':int((d.daily_mean.notna()&~d.daily_mean.between(-10,1000)).sum())})
    for site in d.site_key.unique():
        LOCATIONS.append({'site_key':site,'year':year,'coordinate_status':'undated_api_snapshot_assumption','coordinate_metadata_year':None})
    d=d[d.daily_mean.between(-10,1000)&d.date.dt.year.eq(year)]
    z=monthly(d);z['valid_hours']=np.nan
    z['daily_quality']='official_daily_hours_unavailable';z['time_basis']='official_Date'
    return z

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    record(HERE/'repair_nsw_boundary.py')
    record(HERE/'AMENDMENT_20260930.md')
    data=[]
    for year in (2023,2024,2025):
        for func in (nsw,qld,vic):
            d=func(year);data.append(d)
            print(func.__name__,year,'station_months',len(d),flush=True)
    d=pd.concat(data,ignore_index=True)
    assert not d.duplicated(['site_key','year','month']).any()
    assert (d.valid_days<=d.required_days/.75+1).all()
    loc=pd.DataFrame(LOCATIONS).drop_duplicates()
    assert not loc.duplicated(['site_key','year']).any()
    d=d.merge(loc,on=['site_key','year'],validate='many_to_one')
    poly_path=BASELINE/'data/china_australia_fua.geojson';record(poly_path)
    poly=gpd.read_file(poly_path);poly=poly[poly.Cntry_ISO.eq('AUS')]
    m=d[META].drop_duplicates().reset_index(drop=True);m['location_version']=m.index
    points=gpd.GeoDataFrame(m,geometry=gpd.points_from_xy(m.longitude,m.latitude),crs=4326).to_crs(poly.crs)
    matches=gpd.sjoin(points,poly,how='left',predicate='intersects')
    counts=matches.groupby('location_version').eFUA_ID.count()
    matches['match_count']=matches.location_version.map(counts)
    good=matches[matches.match_count.eq(1)]
    d=d.merge(good[META+['location_version','eFUA_ID']],on=META,how='left',validate='many_to_one')
    d['country_iso']='AUS';d['exposure_status']='previous_outcomes_evaluated_retrospective_only'
    d['historical_role_status']='not_yet_certified_for_extension_year'
    d.to_csv(HERE/'data/australia_station_month_2023_2025.csv.gz',index=False)
    matches.drop(columns='geometry').to_csv(HERE/'audit/extension_polygon_matches.csv',index=False)
    selected=d[d.eligible&d.eFUA_ID.notna()].copy()
    keys=['country_iso','eFUA_ID','year','month']
    fua=selected.groupby(keys).agg(observed_pm25=('observed_pm25','mean'),station_count=('site_key','nunique'),
        sites=('site_key',lambda x:'|'.join(sorted(x)))).reset_index()
    fua['exposure_status']='previous_outcomes_evaluated_retrospective_only'
    fua.to_csv(HERE/'data/australia_fua_month_2023_2025.csv.gz',index=False)
    summary=[]
    for (state,year),g in d.groupby(['network_state','year']):
        valid=g[g.eligible];mapped=valid[valid.eFUA_ID.notna()]
        summary.append({'network_state':state,'year':year,'stations':g.site_key.nunique(),
            'station_months':len(g),'eligible_station_months':len(valid),'mapped_eligible_station_months':len(mapped),
            'mapped_stations':mapped.site_key.nunique(),'fuas':mapped.eFUA_ID.nunique(),
            'missing_coordinate_station_months':int(g.longitude.isna().sum())})
    pd.DataFrame(summary).to_csv(HERE/'audit/extension_coverage.csv',index=False)
    pd.DataFrame(QC).to_csv(HERE/'audit/extension_quality_counts.csv',index=False)
    pd.DataFrame(METHODS).to_csv(HERE/'audit/extension_methods.csv',index=False)
    loc.to_csv(HERE/'audit/extension_coordinate_provenance.csv',index=False)
    pd.DataFrame(UNMATCHED,columns=['network_state','year','station_name']).to_csv(HERE/'audit/unmatched_coordinates.csv',index=False)
    old=pd.read_csv(BASELINE/'data/regulatory_station_month_2015_2022.csv.gz',usecols=META)
    old=old[old.network_state.ne('CNEMC')].drop_duplicates()
    # New IDs are not automatically coalesced with legacy IDs sharing a name.
    def key(x):return re.sub('[^a-z0-9]','',str(x).lower())
    old['name_key']=old.station_name.map(key);m['name_key']=m.station_name.map(key)
    leads=m.merge(old,on=['network_state','name_key'],suffixes=('_new','_old'))
    leads=leads[leads.site_key_new.ne(leads.site_key_old)]
    leads['identity_status']='candidate_only_not_automatically_merged'
    leads.to_csv(HERE/'audit/station_identifier_change_candidates.csv',index=False)
    dump(HERE/'audit/extension_manifest.json',{'created_at':now(),'protocol_sha256':digest(HERE/'PROTOCOL.md'),
        'script_sha256':digest(__file__),'source_files':list({r['path']:r for r in FILES}.values()),
        'station_months':len(d),'stations':int(d.site_key.nunique()),'eligible_station_months':len(d[d.eligible]),
        'fua_months':len(fua),'fuas':int(fua.eFUA_ID.nunique()),'outcome_performance_computed':False,
        'fresh_external_test':False})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('FUA months',len(fua),'FUAs',fua.eFUA_ID.nunique(),flush=True)

if __name__=='__main__':main()
