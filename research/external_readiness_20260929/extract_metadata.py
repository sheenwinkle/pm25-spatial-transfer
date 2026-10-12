"""Year-specific Queensland roles/instruments and Chinese change leads."""
import json
import re
import sys
import pandas as pd
from common import HERE, ROOT, PREVIOUS, digest, dump

def normal(value):return re.sub('[^a-z0-9]','',str(value).lower())

def fields(text):
    matches=list(re.finditer(r'\*\*([^*]+?):\*\*\s*',text))
    return {m.group(1).strip():text[m.end():matches[i+1].start() if i+1<len(matches) else len(text)].strip()
            for i,m in enumerate(matches)}

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    old=ROOT/'data/raw/australia_ground_monitoring/qld/qld_air_quality_ckan_metadata_2015_2024.json'
    catalogs=json.loads(old.read_text(encoding='utf-8'))
    sources={int(y):old for y in catalogs}
    for year in (2024,2025):
        path=HERE/f'evidence/qld_catalogue_{year}.json'
        catalogs[str(year)]=json.loads(path.read_text(encoding='utf-8'))['result'];sources[year]=path
    rows=[]
    for year,cat in catalogs.items():
        for resource in cat['resources']:
            name=resource['name'].split(' (')[0].strip()
            blocks=fields(resource.get('description',''))
            classification=next((v.splitlines()[0].strip() for k,v in blocks.items() if 'site classification' in k.lower()),'')
            coords=blocks.get('Site location','')
            lat=re.search(r'Latitude:\s*(-?\d+\.?\d*)',coords)
            lon=re.search(r'Longitude:\s*(-?\d+\.?\d*)',coords)
            pmblocks=[f'{k}: {v}' for k,v in blocks.items() if 'PM2.5' in k]
            method='\n'.join(pmblocks)
            def method_family(block):
                lower=block.lower()
                if 'scattered light spectrometry' in lower:return 'scattered_light_spectrometry'
                if 'optical light scattering' in lower:return 'optical_light_scattering'
                if 'teom' in lower:
                    if 'excluding fdms' in lower:return 'TEOM_without_FDMS'
                    if 'fdms' in lower:return 'TEOM_FDMS'
                    return 'TEOM_FDMS_not_documented'
                if 'attenuation' in lower:return 'beta_attenuation'
                return 'other_stated_method'
            families=sorted(set(method_family(x) for x in pmblocks))
            family='|'.join(families) if families else 'not_stated_or_not_monitored'
            role=('neighbourhood' if classification.lower().startswith('neighbourhood') else
                'background' if classification.lower().startswith('background') else
                'peak' if classification.lower().startswith('peak') else 'unknown')
            rows.append({'site_key':'QLD_'+normal(name),'station_name':name,'year':int(year),
                'classification_verbatim':classification,'role':role,
                'operation_period':blocks.get('Period of operation',''),
                'latitude':float(lat.group(1)) if lat else None,'longitude':float(lon.group(1)) if lon else None,
                'pm25_method_family':family,'pm25_method_description':method,
                'within_year_method_change':len(families)>1 or bool(re.search(r'\b(until|from) \d',method,re.I)),
                'semivolatile_loss_uncompensated':'do not compensate' in method.lower(),
                'manufacturer_instruction_only':"manufacturer's instruction" in method.lower(),
                'role_evidence_status':'official_year_specific' if role!='unknown' else 'not_stated',
                'source_url':f'https://www.data.qld.gov.au/dataset/{cat["id"]}/resource/{resource["id"]}',
                'catalogue_modified':cat.get('metadata_modified'),'resource_url':resource.get('url'),
                'source_file':str(sources[int(year)]),'source_sha256':digest(sources[int(year)])})
    raw=pd.DataFrame(rows)
    raw.to_csv(HERE/'audit/qld_resource_evidence.csv',index=False)
    keys=['site_key','year'];cols=['classification_verbatim','role','operation_period','latitude','longitude','pm25_method_family','pm25_method_description']
    canonical=[];conflicts=[]
    for key,g in raw.groupby(keys):
        distinct=g[cols].drop_duplicates()
        if len(distinct)>1:
            conflicts.extend(g.to_dict('records'))
            continue
        row=g.iloc[0].to_dict();row['resource_count']=len(g)
        row['all_evidence_urls']='|'.join(sorted(set(g.source_url)))
        canonical.append(row)
    qld=pd.DataFrame(canonical)
    assert not qld.duplicated(keys).any()
    qld.to_csv(HERE/'data/qld_station_year_evidence_2015_2025.csv',index=False)
    pd.DataFrame(conflicts,columns=raw.columns).to_csv(HERE/'audit/qld_conflicting_year_records.csv',index=False)
    changes=[]
    for site,g in qld.sort_values(keys).groupby('site_key'):
        previous=None
        for row in g.to_dict('records'):
            if previous:
                for field in ['role','classification_verbatim','latitude','longitude','pm25_method_family']:
                    if str(previous[field])!=str(row[field]):
                        changes.append({'site_key':site,'field':field,'previous_evidence_year':previous['year'],
                            'next_evidence_year':row['year'],'before':previous[field],'after':row[field],
                            'timing':'between annual snapshots; not exact event date','source_url':row['source_url']})
            previous=row
    pd.DataFrame(changes).to_csv(HERE/'audit/qld_metadata_changes.csv',index=False)
    # Secondary annotations are retained as leads; they do not become verified roles.
    registry=pd.read_csv(PREVIOUS/'audit/station_role_registry.csv')
    china=registry[registry.country_iso.eq('CHN')&registry.membership_all].copy()
    leads=[]
    for row in china.to_dict('records'):
        name=row['station_name_zh']
        notes=re.findall(r'[（(]([^）)]+)[）)]',name)
        for note in notes:
            kind=('closure' if '停运' in note else 'role_change' if '评价点' in note else
                'addition' if '新增' in note else 'control_label' if '对照点' in note else 'other_annotation')
            token=re.search(r'(?<!\d)(\d{8}|\d{6})(?!\d)',note)
            date=None
            if token:
                value=token.group(1)
                try:date=pd.to_datetime(value,format='%Y%m%d' if len(value)==8 else '%y%m%d').date().isoformat()
                except ValueError:pass
            leads.append({'site_key':row['site_key'],'city_zh':row['city_zh'],'parent_city_zh':row['parent_city_zh'],
                'station_name':name,'annotation':note,'event_type':kind,'candidate_event_date':date,
                'evidence_level':'secondary_archive_name_annotation_unverified',
                'source_url':'https://zenodo.org/records/10836299','automatic_role_change_allowed':False})
    pd.DataFrame(leads).to_csv(HERE/'audit/china_historical_event_leads.csv',index=False)
    china[china.role.str.contains('unknown')].to_csv(HERE/'audit/china_unresolved_role_sites.csv',index=False)
    dump(HERE/'audit/metadata_extraction_summary.json',{'qld_resource_records':len(raw),'qld_canonical_station_years':len(qld),
        'qld_conflicting_resource_records':len(conflicts),'qld_metadata_changes':len(changes),
        'china_annotation_leads':len(leads),'china_annotation_sites':len(set(x['site_key'] for x in leads)),
        'china_unmatched_membership_sites':int(china.role.str.contains('unknown').sum()),
        'role_snapshots_backfilled':False})
    print('QLD station-years',len(qld),'conflicting resources',len(conflicts),'China event leads',len(leads),flush=True)

if __name__=='__main__':main()
