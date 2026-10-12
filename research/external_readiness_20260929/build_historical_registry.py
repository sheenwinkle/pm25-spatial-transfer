"""Link report-year evidence to observed station-years without temporal backfill."""
import json
import re
import pandas as pd
from common import HERE, ROOT, PREVIOUS, BASELINE, dump, digest

def normal(value):return re.sub('[^a-z0-9]','',str(value).lower())

def main():
    meta=pd.DataFrame(json.loads((ROOT/'data/raw/australia_ground_monitoring/nsw/nsw_air_quality_site_details.json').read_text(encoding='utf-8')))
    look={normal(r.SiteName):f'NSW_{r.Site_Id}' for r in meta.itertuples()}
    specs={2017:(HERE,[25],22),2018:(HERE,[26,27],28),2019:(PREVIOUS,[23,24],32),
           2020:(HERE,[19,20,21],36),2021:(HERE,[23,24],40),2022:(PREVIOUS,[24,25],44)}
    rows=[];allrows=[]
    for year,(folder,pages,expected) in specs.items():
        path=folder/f'evidence/nsw_nepm_{year}.txt'
        text=path.read_text(encoding='utf-8')
        content=dict((int(n),t) for n,t in re.findall(r'PAGE (\d+)\n(.*?)(?=\nPAGE |\Z)',text,re.S))
        found=[]
        for page in pages:
            for line in content[page].splitlines():
                if year==2017:
                    match=re.match(r'^(.+?)\s+([TPC])\s+(\d{4})\s+\d+\s+X',line)
                    if not match:continue
                    name,classification,start=match.groups();pm=classification
                elif year==2018:
                    match=re.match(r'^\d+\s+(.+?)\s+([TPC])\s+(\d{4})\s+\d+\s+',line)
                    if not match:continue
                    name,classification,start=match.groups();pm=classification
                else:
                    match=re.match(r'^\d+\s+(.+?)\s+(\d{4})\s+([TPC/]+)\s+([TPC/–-]+)(?:\s+([TPC/–-]+))?(?:\s|$)',line)
                    if not match:continue
                    name,start,classification,pm10,pm=match.groups()
                    pm=pm or '-'
                name=re.sub(r'[\s\d,]+$','',name).strip()
                # Explicit presentation aliases supported by each report's context.
                aliases={'Wagga Wagga':'Wagga Wagga North','Cook and Philip':'Cook and Phillip',
                    'Cook and Phillip (Sydney CBD)':'Cook and Phillip','Campbelltown':'Campbelltown West'}
                canonical=aliases.get(name,name)
                key=look.get(normal(canonical))
                found.append({'year':year,'reported_name':name,'station_name':canonical,'site_key':key,
                    'classification_verbatim':classification,'pm25_classification':pm,
                    'reported_start_year':int(start),'pdf_page':page,'source_id':f'nsw_nepm_{year}',
                    'source_file':str(folder/f'evidence/nsw_nepm_{year}.pdf'),
                    'source_sha256':digest(folder/f'evidence/nsw_nepm_{year}.pdf'),
                    'name_alias_applied':name!=canonical,
                    'role':'NEPM_PM25_'+pm if pm not in ['-','–'] else 'PM25_not_listed',
                    'role_evidence_status':'official_report_year_only',
                    'role_harmonisation':'NEPM designation is not proof of pure urban-background exposure'})
        assert len(found)==expected,(year,len(found),expected,[x['station_name'] for x in found])
        assert all(x['site_key'] for x in found),(year,[x for x in found if not x['site_key']])
        allrows.extend(found)
        rows.extend(x for x in found if x['pm25_classification'] not in ['-','–'])
    pd.DataFrame(allrows).to_csv(HERE/'data/nsw_report_year_evidence.csv',index=False)
    source=pd.read_csv(PREVIOUS/'data/station_month_with_roles.csv.gz',low_memory=False)
    # Observational coverage only; no new outcome summaries or prediction selection.
    source=source[source.selected_all].copy()
    grouped=source.groupby(['site_key','country_iso','network_state','station_name','year'],dropna=False).agg(
        observed_months=('month','nunique'),old_role=('role','first'),old_role_selected=('selected_role','max')).reset_index()
    extension=pd.read_csv(HERE/'data/australia_station_month_2023_2025.csv.gz')
    ext=extension[extension.eligible&extension.eFUA_ID.notna()].groupby(
        ['site_key','country_iso','network_state','station_name','year'],dropna=False).agg(observed_months=('month','nunique')).reset_index()
    ext['old_role']='outside_previous_time_window';ext['old_role_selected']=False
    registry=pd.concat([grouped,ext],ignore_index=True)
    registry['annual_role']='unknown'
    registry['annual_role_status']='unverified_for_this_year'
    registry['annual_source']='';registry['annual_source_page']=''
    registry['pm25_method_family']='unverified_for_this_year'
    registry['pm25_method_description']=''
    registry['instrument_caveat']=''
    registry['cross_source_role_conflict_known']=False
    registry['same_year_role_conflict']=False
    qld=pd.read_csv(HERE/'data/qld_station_year_evidence_2015_2025.csv').set_index(['site_key','year'])
    nsw=pd.DataFrame(rows).set_index(['site_key','year'])
    oldreg=pd.read_csv(PREVIOUS/'audit/station_role_registry.csv').set_index('site_key')
    conflicts=[]
    for key,e in qld.iterrows():
        site,year=key
        if site not in oldreg.index or year!=2023:continue
        snapshot=oldreg.loc[site]
        if snapshot.source_id!='qld_nepm_2023' or not str(snapshot.role).startswith('PM25_'):continue
        reported=snapshot.role.split('_')[1]
        if reported!=e.role:
            conflicts.append({'site_key':site,'year':year,'catalogue_AS_NZS_class':e.role,
                'report_population_exposure_class':reported,'catalogue_url':e.source_url,
                'report_source':'qld_nepm_2023','report_pdf_page':78,
                'resolution':'unresolved source/taxonomy discrepancy; neither silently overrides the other'})
    conflict_sites={x['site_key'] for x in conflicts}
    for i,row in registry.iterrows():
        key=(row.site_key,row.year)
        registry.loc[i,'cross_source_role_conflict_known']=row.site_key in conflict_sites
        registry.loc[i,'same_year_role_conflict']=row.site_key in conflict_sites and row.year==2023
        if key in qld.index:
            e=qld.loc[key]
            registry.loc[i,['annual_role','annual_role_status','annual_source','pm25_method_family','pm25_method_description']]=[
                e.role,'official_year_specific',e.source_url,e.pm25_method_family,e.pm25_method_description]
            flags=[]
            if e.semivolatile_loss_uncompensated:flags.append('semivolatile_loss_not_compensated')
            if e.manufacturer_instruction_only:flags.append('manufacturer_instruction_only')
            if e.within_year_method_change:flags.append('within_year_method_change')
            registry.loc[i,'instrument_caveat']='|'.join(flags)
        elif key in nsw.index:
            e=nsw.loc[key]
            registry.loc[i,['annual_role','annual_role_status','annual_source','annual_source_page']]=[
                e.role,'official_report_year_only',e.source_id,str(e.pdf_page)]
            # Network method is not the same as a station-specific instrument log.
            registry.loc[i,'instrument_caveat']='station_specific_instrument_log_pending'
        elif row.site_key in oldreg.index:
            e=oldreg.loc[row.site_key]
            if pd.notna(e.evidence_year) and int(e.evidence_year)==row.year and not str(e.role).startswith('unknown') and e.role!='table_lists_ozone_not_PM25':
                registry.loc[i,['annual_role','annual_role_status','annual_source','annual_source_page']]=[
                    e.role,'official_report_year_only',e.source_id,str(e.pdf_page)]
    assert not registry.duplicated(['site_key','year']).any()
    registry['role_usable_without_backfill']=registry.annual_role_status.ne('unverified_for_this_year')&~registry.same_year_role_conflict
    pd.DataFrame(conflicts).to_csv(HERE/'audit/qld_cross_source_classification_discrepancies.csv',index=False)
    registry.to_csv(HERE/'data/observed_station_year_registry_2015_2025.csv',index=False)
    cols=['site_key','year','annual_role','annual_role_status','annual_source','pm25_method_family','instrument_caveat',
          'cross_source_role_conflict_known','same_year_role_conflict','role_usable_without_backfill']
    annotated=extension.drop(columns=['historical_role_status']).merge(registry[cols],on=['site_key','year'],how='left',validate='many_to_one')
    annotated['annual_role_status']=annotated.annual_role_status.fillna('outside_audited_FUA_cohort')
    annotated.to_csv(HERE/'data/australia_station_month_2023_2025_annotated.csv.gz',index=False)
    summary=registry.groupby(['country_iso','network_state','annual_role_status']).agg(
        station_years=('site_key','size'),stations=('site_key','nunique'),station_months=('observed_months','sum')).reset_index()
    summary.to_csv(HERE/'audit/historical_evidence_coverage.csv',index=False)
    restored=registry[(registry.year<=2022)&registry.network_state.eq('QLD')&~registry.old_role_selected&registry.annual_role.eq('neighbourhood')]
    restored.to_csv(HERE/'audit/previously_unresolved_qld_neighbourhood_years.csv',index=False)
    # Compare classifications only at years with independent annual support.
    qchange=registry[(registry.year<=2022)&registry.network_state.eq('QLD')]
    qchange.to_csv(HERE/'audit/qld_before_after_annual_roles.csv',index=False)
    dump(HERE/'audit/historical_registry_summary.json',{'observed_station_years':len(registry),
        'year_specific_supported_station_years':int(registry.role_usable_without_backfill.sum()),
        'previously_excluded_neighbourhood_station_years':len(restored),
        'previously_excluded_neighbourhood_sites':int(restored.site_key.nunique()),
        'nsw_annual_table_rows':len(allrows),'nsw_pm25_designated_rows':len(rows),
        'qld_same_year_source_class_discrepancies':len(conflicts),
        'china_full_annual_role_verification_complete':False,'old_cohorts_modified':False})
    print(summary.to_string(index=False),flush=True)
    print('Previously excluded QLD neighbourhood years',len(restored),'sites',restored.site_key.nunique(),flush=True)

if __name__=='__main__':main()
