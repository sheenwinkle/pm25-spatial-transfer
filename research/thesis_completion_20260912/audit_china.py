import csv
import io
import zipfile
import sys
import numpy as np
import pandas as pd
from common import ROOT, AUDIT, digest, dump

sys.path.insert(0,str(ROOT/'thesis_workspace'))
from run_national_cnemc_validation import EXPECTED_MD5

rawroot=ROOT/'data/raw/cnemc_zenodo_10836299'
hashes=[];schema=[];checks=[]
for name,expected in EXPECTED_MD5.items():
    path=rawroot/name;actual=digest(path,'md5')
    hashes.append({'name':name,'expected_md5':expected,'observed_md5':actual,'pass':expected==actual})
assert all(r['pass'] for r in hashes)
pd.DataFrame(hashes).to_csv(AUDIT/'china_archive_md5.csv',index=False)
for year in range(2015,2023):
    cached=pd.read_csv(ROOT/f'model_outputs/national_cnemc_validation_20260824/tables/national_cnemc_station_monthly_{year}.csv.gz')
    codes=sorted(cached.station_code.unique());codes=np.array(codes)[np.linspace(0,len(codes)-1,12,dtype=int)]
    daily=[]
    with zipfile.ZipFile(rawroot/f'{year}0101-{year}1231_CNEMC.zip') as z:
        members=sorted(n for n in z.namelist() if n.rsplit('/',1)[-1].startswith('china_sites_') and n.endswith('.csv'))
        for member in members:
            dates=[];hours=[];selected=[]
            with io.TextIOWrapper(z.open(member),encoding='utf-8-sig') as f:
                header=next(csv.reader([f.readline()]));positions={c:header.index(c) for c in codes if c in header}
                for line in f:
                    prefix=line.split(',',3)
                    if len(prefix)<4 or prefix[2].strip()!='PM2.5':continue
                    dates.append(prefix[0].strip());hours.append(int(prefix[1]))
                    if int(prefix[0][4:6]) in (1,7):
                        row=next(csv.reader([line]));selected.append([pd.to_numeric(row[positions[c]],errors='coerce') if c in positions else np.nan for c in codes])
            duplicates=len(hours)-len(set(zip(dates,hours)))
            schema.append({'year':year,'member':member,'pm25_rows':len(hours),'unique_dates':len(set(dates)),
                           'duplicate_date_hour_rows':duplicates,'hour_outside_0_23':sum(h<0 or h>23 for h in hours)})
            assert duplicates==0 and len(set(dates))<=1 and all(0<=h<=23 for h in hours)
            if selected:
                a=np.array(selected,float);a[(a<0)|(a>1000)]=np.nan
                count=np.isfinite(a).sum(0)
                for i,c in enumerate(codes):
                    if count[i]>=20:daily.append({'station_code':c,'month':int(dates[0][4:6]),'mean':np.nanmean(a[:,i]),'hours':count[i]})
    d=pd.DataFrame(daily).groupby(['station_code','month']).agg(raw_mean=('mean','mean'),raw_days=('mean','size'),raw_hours=('hours','sum')).reset_index()
    d=d.merge(cached,on=['station_code','month'],validate='one_to_one');d['difference']=d.raw_mean-d.pm25_mean_ug_m3
    assert np.allclose(d.raw_mean,d.pm25_mean_ug_m3,rtol=0,atol=1e-10)
    assert (d.raw_days==d.valid_days).all() and (d.raw_hours==d.valid_hours).all()
    checks.append(d);print('CHINA_RAW_VERIFIED',year,len(d),flush=True)
pd.DataFrame(schema).to_csv(AUDIT/'china_all_daily_member_schema.csv.gz',index=False)
pd.concat(checks,ignore_index=True).to_csv(AUDIT/'china_raw_monthly_spot_checks.csv',index=False)
dump(AUDIT/'china_raw_verification.json',{'archive_hashes_verified':len(hashes),'daily_members_checked':len(schema),
    'station_months_recomputed':sum(len(d) for d in checks),'max_abs_difference':max(d.difference.abs().max() for d in checks),
    'scope':'Every PM2.5 date-hour header checked; 12 deterministically spaced station codes per year, January/July values reaggregated. Not all values independently reaggregated.'})
