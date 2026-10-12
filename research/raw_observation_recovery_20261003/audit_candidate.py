"""Audit a separately deposited hourly archive before any missing-only merge."""
import hashlib
import io
import json
import re
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from fetch_sources import HERE,dump,sha

ROOT=Path.home()/'.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'


def load_original():
    rows=[]
    path=ROOT/'data/raw/cnemc_zenodo_10836299/20180101-20181231_CNEMC.zip'
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            match=re.search(r'china_sites_(201812\d{2})\.csv$',name)
            if not match:continue
            raw=z.read(name)
            if not raw.startswith(b'date,hour,type,'):continue
            frame=pd.read_csv(io.BytesIO(raw),low_memory=False)
            p=frame[frame.type.astype(str).str.strip().eq('PM2.5')].copy()
            codes=[c for c in p.columns if re.fullmatch(r'\d+[A-Z]',c)]
            time=pd.to_datetime(p.date.astype(str),format='%Y%m%d')+pd.to_timedelta(p.hour,unit='h')
            values=p[codes].apply(pd.to_numeric,errors='coerce');values.index=pd.DatetimeIndex(time)
            rows.append(values)
    wide=pd.concat(rows).sort_index()
    assert not wide.index.duplicated().any()
    return wide.where(wide.ge(0)&wide.le(1000)&np.isfinite(wide)),path


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    path=HERE/'evidence/li_201812_nc.body'
    with open(path,'rb') as f:md5=hashlib.file_digest(f,'md5').hexdigest()
    assert md5=='3f93e6b597a990f6e931631e13b1b517'
    d=xr.open_dataset(path,engine='netcdf4',decode_times=False)
    codes=[x.decode('ascii').strip() for x in d.station_code.values]
    dates=[x.decode('ascii').strip() for x in d.time_point.values]
    assert len(codes)==len(set(codes))
    nonempty=[x for x in dates if x]
    assert len(nonempty)==len(set(nonempty))
    assert d.pm2_5.attrs['units']=='ug m-3'
    stamp=pd.to_datetime(dates,utc=True,errors='coerce').tz_localize(None)
    new=pd.DataFrame(d.pm2_5.values,index=stamp,columns=codes)
    invalid_counts=new.stack().value_counts().loc[lambda x:x.index<0].to_dict()
    new=new.where(new.ge(0)&new.le(1000)&np.isfinite(new))
    invalid_time_rows=new.index.isna()
    unknown_time_valid_values=int(new[invalid_time_rows].notna().sum().sum())
    slot=pd.date_range(d.attrs['start_date'],d.attrs['end_date'],freq='h')
    assert len(slot)==len(stamp)
    valid_timestamp_positions_match_slot=bool((slot[~invalid_time_rows]==stamp[~invalid_time_rows]).all())
    good_slot=(stamp==slot)&~invalid_time_rows
    date_audit=pd.DataFrame({'slot':np.arange(len(stamp)),'expected_slot':slot,'source_timestamp':dates,
        'blank_timestamp':invalid_time_rows,'matches_declared_calendar_slot':good_slot,
        'finite_pm25_values':new.notna().sum(axis=1).to_numpy()})
    date_audit.to_csv(HERE/'audit/li_timestamp_rows.csv',index=False)
    new=new[good_slot]
    old,archive=load_original();common=sorted(set(old.columns)&set(new.columns));diagnostics=[]
    for shift in [-8,0,8]:
        n=new[common].copy();n.index=n.index+pd.Timedelta(hours=shift)
        overlap=old.index.intersection(n.index);a=old.loc[overlap,common].to_numpy();b=n.loc[overlap].to_numpy()
        valid=np.isfinite(a)&np.isfinite(b);e=(a-b)[valid]
        diagnostics.append({'candidate_clock_shift_hours':shift,'overlap_station_hours':len(e),
            'exact_equal':int(np.isclose(e,0,atol=1e-7,rtol=0).sum()),
            'exact_fraction':float(np.isclose(e,0,atol=1e-7,rtol=0).mean()),
            'mae':float(abs(e).mean()),'rmse':float(np.sqrt(np.mean(e**2))),
            'maximum_abs_error':float(abs(e).max())})
    pd.DataFrame(diagnostics).to_csv(HERE/'audit/li_clock_alignment.csv',index=False)
    reference=old.reindex(index=new.index,columns=common);candidate=new[common]
    a=reference.to_numpy();b=candidate.to_numpy()
    missing_only=np.isnan(a)&np.isfinite(b);conflicts=np.isfinite(a)&np.isfinite(b)&~np.isclose(a,b,atol=1e-7,rtol=0)
    extra=[];disagreement=[]
    for i,j in zip(*np.nonzero(missing_only)):
        extra.append({'source_clock':str(new.index[i]),'station_code':common[j],'pm25_ug_m3':b[i,j],
            'status':'candidate_only_clock_convention_unresolved','source_file':'aqi201812.nc'})
    for i,j in zip(*np.nonzero(conflicts)):
        disagreement.append({'source_clock':str(new.index[i]),'station_code':common[j],
            'original_pm25':a[i,j],'candidate_pm25':b[i,j],'absolute_difference':abs(a[i,j]-b[i,j])})
    pd.DataFrame(extra,columns=['source_clock','station_code','pm25_ug_m3','status','source_file']).to_csv(HERE/'data/li_missing_only_candidates.csv',index=False)
    pd.DataFrame(disagreement,columns=['source_clock','station_code','original_pm25','candidate_pm25','absolute_difference']).to_csv(HERE/'audit/li_overlap_conflicts.csv',index=False)
    # Hypothetical only: retain existing values, assess whether candidate additions change eligibility.
    full_index=pd.date_range('2018-12-01','2018-12-31 23:00:00',freq='h')
    original=old.reindex(index=full_index);overlay=original.copy()
    overlay.loc[new.index,common]=reference.combine_first(candidate).to_numpy()
    impacts=[]
    for code in original.columns:
        before=original[code].groupby(original.index.date).count()
        after=overlay[code].groupby(overlay.index.date).count()
        impacts.append({'station_code':code,'valid_days_before':int(before.ge(20).sum()),
            'valid_days_hypothetical':int(after.ge(20).sum()),'extra_valid_hours':int(after.sum()-before.sum()),
            'eligible_before':bool(before.ge(20).sum()>=24),'eligible_hypothetical':bool(after.ge(20).sum()>=24)})
    impacts=pd.DataFrame(impacts);impacts.to_csv(HERE/'audit/li_hypothetical_eligibility.csv',index=False)
    # Keep each day/clock record in source form; diagnostic alignment is not acceptance.
    daily=[]
    for day,g in new.groupby(new.index.date):
        counts=g.notna().sum();means=g.mean()
        for code in codes:daily.append({'date':str(day),'station_code':code,'valid_hours':int(counts[code]),
            'daily_pm25':means[code],'eligible_day':bool(counts[code]>=20)})
    daily=pd.DataFrame(daily);daily.to_csv(HERE/'data/li_201812_source_clock_station_days.csv.gz',index=False)
    daily.groupby('date').agg(station_hours=('valid_hours','sum'),eligible_stations=('eligible_day','sum'),
        max_station_hours=('valid_hours','max')).reset_index().to_csv(HERE/'audit/li_daily_availability.csv',index=False)
    audit={'source_doi':'10.5281/zenodo.6404603','file':'aqi201812.nc','source_sha256':sha(path),
        'md5_verified':True,'site_count':len(codes),'timestamp_count':len(dates),'pm25_units':'ug m-3',
        'negative_value_counts':invalid_counts,'time_strings_end_in_Z':all(x.endswith('Z') for x in nonempty),
        'blank_timestamp_rows':int(invalid_time_rows.sum()),'valid_values_without_timestamp':unknown_time_valid_values,
        'valid_timestamp_positions_match_slot':valid_timestamp_positions_match_slot,
        'nonblank_timestamp_slot_mismatches':int((~good_slot&~invalid_time_rows).sum()),
        'quarantined_finite_values_bad_time':int(date_audit.loc[~good_slot,'finite_pm25_values'].sum()),
        'valid_timestamps_used':int(good_slot.sum()),'missing_only_candidate_station_hours':len(extra),
        'overlap_conflicts':len(disagreement),'new_eligible_station_months_hypothetical':int((~impacts.eligible_before&impacts.eligible_hypothetical).sum()),
        'added_eligible_station_days_hypothetical':int((impacts.valid_days_hypothetical-impacts.valid_days_before).sum()),
        'station_code_intersection':len(common),'station_codes_only_in_candidate':sorted(set(codes)-set(old.columns)),
        'original_archive_sha256':sha(archive),'clock_tests':diagnostics,
        'metadata':d.attrs,'candidate_is_independent_instrument_network':False,'merged_into_model_data':False}
    dump(HERE/'audit/li_candidate_audit.json',audit)
    print(pd.DataFrame(diagnostics).to_string(index=False),flush=True)
    print(daily.groupby('date').valid_hours.agg(['sum','max']).to_string(),flush=True)


if __name__=='__main__':main()
