"""Separate retained archive records, quality eligibility and fitted panels."""
import calendar
import pandas as pd
from run_gap_models import HERE,CORRECTED,read,dump,sha


def main():
    path=CORRECTED/'data/station_month_with_roles.csv.gz'
    d=read(path,low_memory=False)
    expected=[int(__import__('math').ceil(.75*calendar.monthrange(int(y),int(m))[1])) for y,m in zip(d.year,d.month)]
    assert (d.required_days.to_numpy()==expected).all()
    assert d.eligible.eq(d.valid_days.ge(d.required_days)&d.observed_pm25.notna()).all()
    rows=[]
    for scope,mask in [('all_archive_rows',pd.Series(True,index=d.index)),('regional_membership',d.membership_all),('regional_role',d.documented_role)]:
        for (country,network),g in d[mask].groupby(['country_iso','network_state']):
            rows.append({'scope':scope,'country':country,'network':network,'retained_station_month_rows':len(g),
                'eligible_rows':int(g.eligible.sum()),'ineligible_rows':int((~g.eligible).sum()),
                'station_identifiers':g.site_key.nunique(),'median_valid_days':g.valid_days.median(),
                'ineligible_days_shortfall_median':(g.required_days-g.valid_days)[~g.eligible].median(),
                'unverifiable_daily_hour_count_rows':int(g.daily_quality.eq('official_daily_hours_unavailable').sum())})
    pd.DataFrame(rows).to_csv(HERE/'audit/station_month_coverage.csv',index=False)
    yearly=d.groupby(['country_iso','network_state','year']).agg(retained_station_months=('site_key','size'),eligible_station_months=('eligible','sum')).reset_index()
    yearly.to_csv(HERE/'audit/station_month_coverage_by_year.csv',index=False)
    dump(HERE/'audit/coverage_inventory_receipt.json',{'input':str(path),'sha256':sha(path),
        'rows':len(d),'calendar_threshold_verified':True,'denominator':'retained archive station-month records, not all expected station-months',
        'nsw_daily_hour_counts_verified':False,'absent_station_months_counted':False})
    print(pd.DataFrame(rows).to_string(index=False),flush=True)


if __name__=='__main__':main()
