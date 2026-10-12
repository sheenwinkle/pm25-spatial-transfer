"""Build a provenance-preserving regulatory FUA-month cohort from local archives."""
import calendar
import gzip
import io
import re
import zipfile
import numpy as np
import pandas as pd
import geopandas as gpd
import joblib
from common import ROOT, HERE, MONTHLY, DATA, AUDIT, dump, digest

AU = ROOT / 'data/raw/australia_ground_monitoring'
QC = []
FILES = []
META = ['site_key', 'station_name', 'network_state', 'longitude', 'latitude']

def record(p):
    FILES.append({'path': str(p), 'bytes': p.stat().st_size, 'sha256': digest(p)})

def monthly(d):
    d = d.copy()
    d['year'] = d.date.dt.year
    d['month'] = d.date.dt.month
    z = d.groupby(META+['year', 'month'], dropna=False).agg(
        observed_pm25=('daily_mean', 'mean'), valid_days=('daily_mean', 'count'),
        valid_hours=('valid_hours', 'sum')).reset_index()
    z['required_days'] = [int(np.ceil(.75*calendar.monthrange(int(y), int(m))[1])) for y,m in zip(z.year,z.month)]
    z['eligible'] = z.valid_days >= z.required_days
    return z

def aggregate(h, tag, shift=0):
    h = h.copy()
    h['value'] = pd.to_numeric(h.value, errors='coerce')
    h['datetime'] = pd.to_datetime(h.datetime, errors='coerce') + pd.Timedelta(hours=shift)
    finite = h.value.notna()
    QC.append({'source': tag, 'shift_hours': shift, 'raw_rows':len(h),
               'finite':int(finite.sum()), 'negative':int((h.value<0).sum()),
               'outside_minus10_1000':int((finite&~h.value.between(-10,1000)).sum()),
               'duplicate_site_hours':int(h.duplicated(['site_key','datetime']).sum())})
    h = h[h.value.between(-10,1000)&h.datetime.notna()].copy()
    # If multiple methods coexist, consistently prefer STP-corrected BAM, then BAM.
    h['rank'] = h.method.map({'SBPM25':0,'BPM2.5':1,'PM2.5':2}).fillna(3)
    h = h.sort_values('rank', kind='stable').drop_duplicates(['site_key','datetime'])
    h['date'] = h.datetime.dt.floor('D')
    d = h.groupby(META+['date'],dropna=False).agg(daily_mean=('value','mean'),valid_hours=('value','count')).reset_index()
    d = d[d.valid_hours>=20]
    return monthly(d)

def cn():
    folder = ROOT/'model_outputs/national_cnemc_validation_20260824/tables'
    parts=[]
    for year in range(2015,2023):
        p=folder/f'national_cnemc_station_monthly_{year}.csv.gz';record(p)
        d=pd.read_csv(p)
        d=d.rename(columns={'pm25_mean_ug_m3':'observed_pm25'})
        d['site_key']='CHN_'+d.station_code.astype(str);d['network_state']='CNEMC'
        d['required_days']=[int(np.ceil(.75*calendar.monthrange(year,int(m))[1])) for m in d.month]
        d['eligible']=d.valid_days>=d.required_days
        d['daily_quality']='20h_cached_raw_spot_checked';d['time_basis']='archive_date_hour'
        parts.append(d[META+['year','month','observed_pm25','valid_days','valid_hours','required_days','eligible','daily_quality','time_basis']])
    return pd.concat(parts,ignore_index=True)

def qld():
    def normal(v):return re.sub('[^a-z0-9]','',str(v).lower())
    p=AU/'qld/qld_air_quality_station_year_metadata_2015_2024.csv';record(p)
    meta=pd.read_csv(p);meta['key']=meta.station_name.map(normal)
    fallback=pd.read_csv(AU/'qld/monitoring-sites-on-open-data.csv')
    fallback=fallback[fallback['Monitoring type'].eq('Air quality')].copy()
    fallback['key']=fallback['Site name'].map(normal)
    parts=[];alternatives=[];unmatched=[]
    for year in range(2015,2023):
        p=AU/f'qld/hourly_pm25/pm2-5-qld-{year}.csv';record(p)
        raw=pd.read_csv(p,low_memory=False)
        dt=pd.to_datetime(raw.Date.astype(str)+' '+raw.Time.astype(str),dayfirst=True,errors='coerce')
        frames=[]
        for col in raw.columns[2:]:
            name=re.sub(r'\s*\(ug/m\^3\)\s*$','',col).split(' (')[0].strip();key=normal(name)
            m=meta[meta.key.eq(key)].copy()
            if len(m):
                m['delta']=(m.year-year).abs();row=m.sort_values(['delta','year','resource_name']).iloc[0]
                lon,lat=row.station_longitude,row.station_latitude
            else:
                m=fallback[fallback.key.eq(key)]
                if not len(m):unmatched.append({'year':year,'name':name});continue
                row=m.iloc[-1];lon,lat=row.Longitude,row.Latitude
            frames.append(pd.DataFrame({'datetime':dt,'value':raw[col], 'site_key':'QLD_'+key,
                'station_name':name,'network_state':'QLD','longitude':lon,'latitude':lat,'method':'PM2.5'}))
        h=pd.concat(frames,ignore_index=True)
        parts.append(aggregate(h,f'QLD_{year}'))
        alternatives.append(aggregate(h,f'QLD_{year}',10))
        print('QLD',year,flush=True)
    d=pd.concat(parts,ignore_index=True);d['daily_quality']='20h_recomputed';d['time_basis']='native_archive_day_timezone_unconfirmed'
    a=pd.concat(alternatives,ignore_index=True);a.to_csv(AUDIT/'qld_plus10_hour_monthly.csv.gz',index=False)
    pd.DataFrame(unmatched).to_csv(AUDIT/'qld_unmatched_metadata.csv',index=False)
    return d

def vic():
    parts=[]
    for year in range(2015,2023):
        cache=DATA/f'vic_recomputed_{year}.csv.gz'
        p=AU/f'vic/hourly_airwatch/{year}_All_sites_air_quality_hourly_avg.xlsx';record(p)
        # Annual caches are private to this branch; original intermediates stay immutable.
        if cache.exists():parts.append(pd.read_csv(cache));continue
        x=pd.ExcelFile(p,engine='calamine');frames=[]
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
            for sheet in x.sheet_names:
                if sheet in ['Metadata','Data Info Panel','AllData']:continue
                raw=pd.read_excel(x,sheet_name=sheet)
                methods=[c for c in ['SBPM25','BPM2.5','PM2.5'] if c in raw]
                if not methods or raw.empty:continue
                sid=int(raw.location_id.dropna().iloc[0]);row=loc.loc[sid]
                dtcol='datetime_local' if 'datetime_local' in raw else 'datetime_AEST'
                for method in methods:
                    frames.append(pd.DataFrame({'datetime':raw[dtcol],'value':raw[method], 'method':method,
                        'site_key':f'VIC_{sid}','station_name':row['name'],'network_state':'VIC',
                        'longitude':float(row.lon),'latitude':float(row.lat)}))
            basis='native_datetime_local_or_AEST_column'
        d=aggregate(pd.concat(frames,ignore_index=True),f'VIC_{year}')
        d['daily_quality']='20h_recomputed';d['time_basis']=basis
        d=d[d.year.eq(year)];d.to_csv(cache,index=False);parts.append(d)
        print('VIC',year,len(d),flush=True)
    return pd.concat(parts,ignore_index=True)

def nsw():
    p=AU/'nsw/nsw_air_quality_site_details.json';record(p)
    import json
    meta=pd.DataFrame(json.loads(p.read_text(encoding='utf-8')))
    parts=[]
    for year in range(2015,2023):
        p=AU/f'nsw/daily_pm25/nsw_pm25_daily_{year}.json.gz';record(p)
        with gzip.open(p,'rt',encoding='utf-8') as f:raw=pd.DataFrame(json.load(f))
        raw=raw.merge(meta[['Site_Id','SiteName','Longitude','Latitude']],on='Site_Id',validate='many_to_one')
        d=raw.rename(columns={'SiteName':'station_name','Longitude':'longitude','Latitude':'latitude','Value':'daily_mean'})
        d['daily_mean']=pd.to_numeric(d.daily_mean,errors='coerce');d['date']=pd.to_datetime(d.Date)
        d['site_key']='NSW_'+d.Site_Id.astype(str);d['network_state']='NSW';d['valid_hours']=np.nan
        assert not d.duplicated(['site_key','date']).any()
        d=d[d.daily_mean.between(-10,1000)]
        z=monthly(d);z['valid_hours']=np.nan
        z['daily_quality']='official_daily_hours_unavailable';z['time_basis']='official_Date'
        parts.append(z)
    return pd.concat(parts,ignore_index=True)

def spatial(d):
    p=ROOT/'data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip';record(p)
    with zipfile.ZipFile(p) as z:member=next(n for n in z.namelist() if n.endswith('.gpkg'))
    poly=gpd.read_file('/vsizip/'+p.as_posix()+'/'+member)
    poly=poly[poly.Cntry_ISO.isin(['CHN','AUS'])].copy();poly['eFUA_ID']=poly.eFUA_ID.astype(int)
    poly=poly[['eFUA_ID','Cntry_ISO','eFUA_name','geometry']]
    m=d[META].drop_duplicates().reset_index(drop=True);m['location_version']=m.index
    points=gpd.GeoDataFrame(m,geometry=gpd.points_from_xy(m.longitude,m.latitude),crs=4326).to_crs(poly.crs)
    matches=gpd.sjoin(points,poly,how='left',predicate='intersects')
    counts=matches.groupby('location_version').eFUA_ID.count();matches['match_count']=matches.location_version.map(counts)
    matches['primary_match']=matches.match_count.eq(1)
    matches['country_expected']=np.where(matches.network_state.eq('CNEMC'),'CHN','AUS')
    matches['primary_match'] &= matches.Cntry_ISO.eq(matches.country_expected)
    matches.drop(columns='geometry').to_csv(AUDIT/'station_polygon_matches.csv',index=False)
    good=matches[matches.primary_match].drop(columns=['geometry','index_right'])
    out=d.merge(good[META+['location_version','eFUA_ID']],on=META,validate='many_to_one',how='left')
    out['country_iso']=np.where(out.network_state.eq('CNEMC'),'CHN','AUS')
    poly.to_crs(4326).to_file(DATA/'china_australia_fua.geojson',driver='GeoJSON')
    return out

def main():
    d=pd.concat([cn(),qld(),vic(),nsw()],ignore_index=True)
    assert not d.duplicated(['site_key','year','month']).any()
    d=spatial(d)
    d.to_csv(DATA/'regulatory_station_month_2015_2022.csv.gz',index=False)
    pd.DataFrame(QC).to_csv(AUDIT/'hourly_quality_counts.csv',index=False)
    raw=d[d.eligible&d.eFUA_ID.notna()].copy()
    keys=['country_iso','eFUA_ID','year','month']
    panel,features,_=joblib.load(MONTHLY/'modelling_cache.joblib')
    panel=panel.rename(columns={'pm25_mean_ug_m3':'satpm_pm25'})
    for cohort,mask in [('all_networks',pd.Series(True,index=raw.index)),('hour_auditable',raw.network_state.ne('NSW'))]:
        z=raw[mask].groupby(keys).agg(observed_pm25=('observed_pm25','mean'),
            station_count=('site_key','nunique'),station_sd=('observed_pm25','std'),
            sites=('site_key',lambda s:'|'.join(sorted(s)))).reset_index()
        out=z.merge(panel,on=keys,how='left',validate='one_to_one',indicator=True)
        missing=out[out._merge.ne('both')]
        missing.to_csv(AUDIT/f'{cohort}_unmatched_model_fuas.csv',index=False)
        out=out[out._merge.eq('both')].drop(columns='_merge')
        out.to_csv(DATA/f'{cohort}_model_panel.csv.gz',index=False)
        summary=out.groupby('country_iso').agg(rows=('observed_pm25','size'),fuas=('eFUA_ID','nunique'),
            grids=('weather_grid_id','nunique'),years=('year','nunique'),mean_pm25=('observed_pm25','mean'))
        summary.to_csv(AUDIT/f'{cohort}_summary.csv')
        print(cohort,summary.to_string(),flush=True)
    dump(AUDIT/'raw_and_cache_manifest.json',FILES)
    dump(AUDIT/'build_summary.json',{'station_months':len(d),'stations':d.site_key.nunique(),
        'valid_station_months':int(d.eligible.sum()),'polygon_matched_valid_station_months':len(raw),
        'unmatched_station_months':int(d.eFUA_ID.isna().sum()),'features':features['monthly_common'],
        'protocol_sha256':digest(HERE/'STATUS.md'),'script_sha256':digest(__file__)})

if __name__=='__main__':main()
