"""Rebuild monthly observed targets under official membership and role snapshots."""
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'thesis_completion_20260912'
FOCUSED = HERE.parent / 'focused_conditions_20260919'
ROOT = Path.home() / '.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'
sys.path.insert(0, str(FOCUSED))
from run_focused import load_panel, dump, sha, read

KEYS = ['country_iso', 'eFUA_ID', 'year', 'month']
COHORTS = ['membership_all', 'documented_role']
COUNTY = {'江阴': '无锡', '宜兴': '无锡', '常熟': '苏州', '张家港': '苏州',
          '昆山': '苏州', '吴江': '苏州', '太仓': '苏州', '海门': '南通',
          '金坛': '常州', '溧阳': '常州', '句容': '镇江', '临安': '杭州',
          '富阳': '杭州', '义乌': '金华', '诸暨': '绍兴'}
JS_URL = 'https://jszd.stats.gov.cn/gjtjjjsdczd/tjbz/content/d0361b20-928c-48ad-ad25-3a003f4073ec.html'
ZJ_URL = 'https://zjjcmspublic.oss-cn-hangzhou-zwynet-d01-a.internet.cloud.zj.gov.cn/jcms_files/jcms1/web3077/site/flash/tjj/Reports1/2016年统计年鉴光盘20190415/html/cn/二、人口和就业人员/2-2 各市、县总户数和总人口数(2015年底).html'


def canonical(s):
    return re.sub(r'\s+', '', str(s))


def station_registry(raw):
    cities = json.loads((HERE / 'data/yrd26_city_names.json').read_text(encoding='utf-8'))
    meta_path = ROOT / 'data/raw/cnemc_zenodo_10836299/Station_information.csv'
    meta = pd.read_csv(meta_path, encoding='gb18030')
    dump(HERE / 'audit/metadata_duplicate_check.json', {'exact_duplicate_rows': int(meta.duplicated().sum()),
         'action': 'collapse only entirely identical metadata rows; no measurements removed'})
    meta = meta.drop_duplicates()
    meta['site_key'] = 'CHN_' + meta.Code.astype(str).str.strip()
    meta = meta[meta.site_key.isin(raw.site_key.unique())].copy()
    assert not meta.site_key.duplicated().any()
    meta['city_zh'] = meta.City_Chinese.map(canonical)
    meta['station_name_zh'] = meta.Station_name_Chinese.map(canonical)
    meta['parent_city_zh'] = meta.city_zh.replace(COUNTY)
    roster = read(HERE / 'data/china_official_roles_2012.csv')
    meta = meta.merge(roster, on=['city_zh', 'station_name_zh'], how='left', validate='many_to_one')
    meta['membership_all'] = meta.parent_city_zh.isin(cities)
    meta['documented_role'] = meta.membership_all & meta.role.eq('network_regular_2012')
    meta['role'] = meta.role.fillna('unknown_unmatched_2012_roster')
    meta['role_stability_assumed'] = meta.evidence_year.notna()
    meta['role_validity'] = np.where(meta.evidence_year.notna(), '2012 designation snapshot; not annually verified 2015-2022', 'unknown')
    meta['geography_evidence'] = 'yrd_plan_2016'
    meta.loc[meta.city_zh.isin(COUNTY), 'geography_evidence'] = 'yrd_plan_2016 + explicit county-parent lookup'
    meta['country_iso'] = 'CHN'
    meta['match_method'] = 'exact Chinese administrative city and whitespace-normalized station name; no fuzzy aliases'
    cols = ['site_key', 'country_iso', 'city_zh', 'parent_city_zh', 'station_name_zh',
            'membership_all', 'documented_role', 'role', 'evidence_year', 'pdf_page',
            'source_id', 'role_stability_assumed', 'role_validity', 'geography_evidence', 'match_method']
    cn = meta[cols].copy()
    au = raw[raw.country_iso.eq('AUS')][['site_key', 'country_iso']].drop_duplicates()
    for c in cols:
        if c not in au:
            au[c] = None
    au['membership_all'] = True
    au['documented_role'] = False
    au['role'] = 'unknown_not_documented_in_selected_rosters'
    au['role_stability_assumed'] = False
    au['role_validity'] = 'unknown'
    au['geography_evidence'] = 'existing unique same-country GHS-FUA point assignment'
    au['match_method'] = 'explicit archived station ID and name crosswalk'

    def assign(ids, role, source, year, page, included):
        found = au.site_key.isin(ids)
        au.loc[found, ['role', 'source_id', 'evidence_year', 'pdf_page', 'documented_role',
                       'role_stability_assumed', 'role_validity']] = [
            role, source, year, page, included, True,
            f'{year} designation snapshot; not annually verified 2015-2022']

    nsw_ids = [1001, 107, 113, 1141, 1148, 15, 155, 1750, 206, 2560, 2570,
               264, 287, 2921, 300, 322, 33, 39, 500, 526, 573, 760, 919]
    assign([f'NSW_{n}' for n in nsw_ids], 'NEPM_not_designated_roadside_2022', 'nsw_nepm_2022', 2022, '24-25', True)
    assign(['NSW_61'], 'designated_roadside', 'nsw_nepm_2022', 2022, '23,26', False)
    assign(['NSW_294', 'NSW_295', 'NSW_304'], 'Newcastle_local_industrial_purpose', 'nsw_newcastle_network', 2021, None, False)
    neighbourhood = ['coomera', 'deagon', 'deceptionbay', 'flindersview', 'mountaincreek',
                     'parkwood', 'rocklea', 'southport', 'springwood', 'toowoomba', 'maryborough',
                     'southgladstone', 'boyneisland', 'westmackay', 'northward', 'woree',
                     'thegap', 'blackwater', 'emerald']
    assign(['QLD_' + n for n in neighbourhood], 'PM25_neighbourhood_2023', 'qld_nepm_2023', 2023, 78, True)
    assign(['QLD_' + n for n in ['brisbanecbd', 'cannonhill', 'southbrisbane', 'woolloongabba', 'boatcreek', 'clinton']],
           'PM25_peak_2023', 'qld_nepm_2023', 2023, 78, False)
    assign(['QLD_' + n for n in ['northmaclean', 'mutdapilly', 'targinie']],
           'PM25_background_2023', 'qld_nepm_2023', 2023, 78, False)
    assign(['VIC_10001', 'VIC_10003', 'VIC_10451'], 'NEPM_PM25_residential_upper_bound_2021', 'vic_nepm_2021', 2021, 6, True)
    assign(['VIC_10107'], 'NEPM_PM25_industrial_residential_upper_bound_2021', 'vic_nepm_2021', 2021, 6, True)
    assign(['VIC_10239'], 'NEPM_PM25_commercial_residential_upper_bound_2021', 'vic_nepm_2021', 2021, 6, True)
    # Melton is listed for ozone, not PM2.5, in table 1. It is not an eligible match.
    assign(['VIC_10169', 'VIC_10437'], 'table_lists_ozone_not_PM25', 'vic_nepm_2021', 2021, 6, False)
    au['membership_all'] = au.membership_all.astype(bool)
    au['documented_role'] = au.documented_role.astype(bool)
    registry = pd.concat([cn, au[cols]], ignore_index=True)
    assert not registry.site_key.duplicated().any()
    missing = set(raw.site_key) - set(registry.site_key)
    assert not missing, sorted(missing)
    pd.DataFrame([{'archive_city': k, 'parent_city': v,
                   'source_url': ZJ_URL if k in ['临安', '富阳', '义乌', '诸暨'] else JS_URL,
                   'evidence_year': 2015 if k in ['临安', '富阳', '义乌', '诸暨'] else 2013}
                  for k, v in COUNTY.items()]).to_csv(HERE / 'audit/county_parent_crosswalk.csv', index=False, encoding='utf-8-sig')
    registry.to_csv(HERE / 'audit/station_role_registry.csv', index=False, encoding='utf-8-sig')
    return registry


def main():
    for folder in ['data', 'audit']:
        (HERE / folder).mkdir(exist_ok=True)
    raw_path = OLD / 'data/regulatory_station_month_2015_2022.csv.gz'
    raw = read(raw_path)
    assert not raw.duplicated(['site_key', 'year', 'month']).any()
    registry = station_registry(raw)
    d = raw.merge(registry.drop(columns='country_iso'), on='site_key', validate='many_to_one')
    d['base_eligible'] = d.eligible & d.eFUA_ID.notna()
    d['selected_all'] = d.base_eligible & d.membership_all
    d['selected_role'] = d.base_eligible & d.documented_role
    d.to_csv(HERE / 'data/station_month_with_roles.csv.gz', index=False)
    # Reuse covariates, never old all-station targets for the restricted cohort.
    base = load_panel()
    covariates = base.drop(columns=['observed_pm25', 'station_count', 'station_sd', 'sites'])
    panels, summaries = {}, []
    for cohort, selection in [('membership_all', 'selected_all'), ('documented_role', 'selected_role')]:
        z = d[d[selection]].copy()
        aggregate = z.groupby(KEYS).agg(observed_pm25=('observed_pm25', 'mean'),
                        station_count=('site_key', 'nunique'), station_sd=('observed_pm25', 'std'),
                        sites=('site_key', lambda s: '|'.join(sorted(s)))).reset_index()
        panel = aggregate.merge(covariates, on=KEYS, how='left', validate='one_to_one', indicator=True)
        assert panel._merge.eq('both').all()
        panel = panel.drop(columns='_merge').sort_values(KEYS).reset_index(drop=True)
        assert not panel.duplicated(KEYS).any()
        panel.to_csv(HERE / f'data/{cohort}_panel.csv.gz', index=False)
        panels[cohort] = panel
        for country, c in panel.groupby('country_iso'):
            stations = z[z.country_iso.eq(country)]
            summaries.append({'cohort': cohort, 'country_iso': country, 'station_months': len(stations),
                              'stations': stations.site_key.nunique(), 'fua_months': len(c),
                              'fuas': c.eFUA_ID.nunique(), 'weather_grids': c.weather_grid_id.nunique(),
                              'first_year': c.year.min(), 'last_year': c.year.max(),
                              'prefectures': stations.parent_city_zh.nunique(),
                              'unknown_role_station_months': int(stations.role.str.startswith('unknown').sum())})
        panel.groupby(['country_iso', 'year']).agg(rows=('observed_pm25', 'size'),
            fuas=('eFUA_ID', 'nunique')).reset_index().assign(cohort=cohort).to_csv(HERE / f'audit/{cohort}_annual_support.csv', index=False)
    pd.DataFrame(summaries).to_csv(HERE / 'audit/cohort_summary.csv', index=False)
    paired = panels[COHORTS[0]][KEYS + ['observed_pm25', 'station_count']].merge(
        panels[COHORTS[1]][KEYS + ['observed_pm25', 'station_count']], on=KEYS,
        suffixes=('_all', '_role'), how='outer', indicator=True, validate='one_to_one')
    paired['target_delta'] = paired.observed_pm25_role - paired.observed_pm25_all
    paired.to_csv(HERE / 'audit/target_changes.csv.gz', index=False)
    # The official scope may include FUAs beyond the prior rectangle, and omit others.
    cn = base[base.country_iso.eq('CHN')]
    rect = cn.fua_centroid_lon.between(117, 123) & cn.fua_centroid_lat.between(28, 34)
    geography = cn[['eFUA_ID', 'fua_name', 'fua_centroid_lon', 'fua_centroid_lat']].drop_duplicates()
    geography['old_east_rectangle'] = geography.eFUA_ID.isin(cn.loc[rect, 'eFUA_ID'])
    geography['new_official_membership'] = geography.eFUA_ID.isin(panels[COHORTS[0]].query("country_iso == 'CHN'").eFUA_ID)
    geography[geography.old_east_rectangle | geography.new_official_membership].to_csv(HERE / 'audit/geography_before_after.csv', index=False)
    splits = {}
    for country in ['CHN', 'AUS']:
        eligible = []
        for panel in panels.values():
            n = panel[panel.country_iso.eq(country) & panel.year.eq(2018)].groupby('eFUA_ID').size()
            eligible.append(set(n[n >= 6].index))
        common = sorted(set.intersection(*eligible))
        assert len(common) >= 6, (country, common)
        splits[country] = {'eligible_common_2018': common, 'seeds': {}}
        for seed in [5993, 5994, 5995]:
            ids = np.array(common)
            np.random.default_rng(seed).shuffle(ids)
            splits[country]['seeds'][str(seed)] = ids[:min(10, int(np.ceil(len(ids) / 2)))].tolist()
    dump(HERE / 'audit/fixed_city_splits.json', splits)
    d[d.selected_all].groupby(['country_iso', 'role'], dropna=False).agg(
        stations=('site_key', 'nunique'), station_months=('site_key', 'size'),
        kept_role_months=('selected_role', 'sum')).reset_index().to_csv(HERE / 'audit/role_coverage.csv', index=False)
    dump(HERE / 'audit/build_receipt.json', {'protocol_sha256': sha(HERE / 'PROTOCOL.md'),
          'script_sha256': sha(__file__), 'inputs': {str(raw_path): sha(raw_path),
          str(OLD / 'data/all_networks_model_panel.csv.gz'): sha(OLD / 'data/all_networks_model_panel.csv.gz'),
          str(OLD / 'data/common_era5_month_2015_2022.csv.gz'): sha(OLD / 'data/common_era5_month_2015_2022.csv.gz')},
          'summaries': summaries, 'classification_is_year_verified': False,
          'official_boundary_polygon_acquired': False})
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print('Shared eligible city counts', {c: len(v['eligible_common_2018']) for c, v in splits.items()}, flush=True)


if __name__ == '__main__':
    main()
