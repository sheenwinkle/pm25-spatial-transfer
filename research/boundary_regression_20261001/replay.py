"""Frozen-rule replay: change NSW acquisition coverage, not model selection."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import calendar
import gzip
import json
import sys
from pathlib import Path
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'regional_station_scope_20260920'
READINESS=HERE.parent/'external_readiness_20260929'
BASE=HERE.parent/'thesis_completion_20260912'
ROOT=Path.home()/'.codex/.chatgpt-projects/g-p-69aaacd3007481919301f520a145f3f4'
sys.path.insert(0,str(PREVIOUS))
import run_regional as original
from run_focused import rulemask, score, support, metrics

KEYS=['country_iso','eFUA_ID','year','month']
SKEYS=['site_key','year','month']
TARGETS=['observed_pm25','station_count','station_sd','sites']
for name in ['data','audit','results']:(HERE/name).mkdir(exist_ok=True)
read=original.read
dump=original.dump
sha=original.sha

def stamp():return datetime.now(timezone.utc).isoformat()

def aggregate(d):
    return d.groupby(KEYS).agg(observed_pm25=('observed_pm25','mean'),station_count=('site_key','nunique'),
                              station_sd=('observed_pm25','std'),sites=('site_key',lambda x:'|'.join(sorted(x)))).reset_index()

def replace_targets(old,z,name):
    out=z.merge(old.drop(columns=TARGETS),on=KEYS,how='left',validate='one_to_one',indicator=True)
    assert out._merge.eq('both').all(),(name,'new month lacks old covariates')
    # Preserve the original row order where it existed, including national source fits.
    order=old[KEYS].assign(_order=np.arange(len(old)))
    out=out.drop(columns='_merge').merge(order,on=KEYS,how='left',validate='one_to_one').sort_values('_order').drop(columns='_order')
    compare=old[KEYS+TARGETS].merge(out[KEYS+TARGETS],on=KEYS,how='outer',suffixes=('_old','_new'),indicator=True)
    compare['target_delta']=compare.observed_pm25_new-compare.observed_pm25_old
    compare.to_csv(HERE/f'audit/{name}_target_changes.csv.gz',index=False)
    out=out[old.columns].reset_index(drop=True)
    # Shared covariates must be byte-equivalent numerical values after a round-trip read.
    common=old.merge(out,on=KEYS,suffixes=('_old','_new'),validate='one_to_one')
    for col in old.columns:
        if col in KEYS+TARGETS:continue
        a,b=common[col+'_old'],common[col+'_new']
        assert (a.eq(b)|(a.isna()&b.isna())).all(),(name,col)
    out.to_csv(HERE/f'data/{name}_panel.csv.gz',index=False)
    return out,compare

def prepare():
    baseline=read(BASE/'data/regulatory_station_month_2015_2022.csv.gz')
    ns=baseline[baseline.network_state.eq('NSW')]
    stable=[c for c in baseline.columns if c not in ['year','month','observed_pm25','valid_days','valid_hours','required_days','eligible']]
    metadata=ns[stable].drop_duplicates()
    assert not metadata.site_key.duplicated().any()
    parts=[]
    for year in range(2015,2023):
        with gzip.open(ROOT/f'data/raw/australia_ground_monitoring/nsw/daily_pm25/nsw_pm25_daily_{year}.json.gz','rt',encoding='utf-8') as f:
            daily=pd.DataFrame(json.load(f))
        patch=pd.DataFrame(json.loads((READINESS/f'data/nsw_boundary_raw/nsw_year_end_{year}.json').read_text(encoding='utf-8')))
        patch=patch[patch.Date.eq(f'{year}-12-31')]
        daily=pd.concat([daily,patch],ignore_index=True)
        assert not daily.duplicated(['Site_Id','Date']).any()
        assert daily.groupby('Site_Id').Date.nunique().eq(366 if calendar.isleap(year) else 365).all()
        daily['value']=pd.to_numeric(daily.Value,errors='coerce')
        daily=daily[daily.value.between(-10,1000)].copy()
        daily['site_key']='NSW_'+daily.Site_Id.astype(str)
        daily['year']=year;daily['month']=pd.to_datetime(daily.Date).dt.month
        z=daily.groupby(SKEYS).agg(observed_pm25=('value','mean'),valid_days=('value','count')).reset_index()
        z['required_days']=[int(np.ceil(.75*calendar.monthrange(year,int(m))[1])) for m in z.month]
        z['eligible']=z.valid_days>=z.required_days;z['valid_hours']=np.nan
        z=z.merge(metadata,on='site_key',how='left',validate='many_to_one')
        assert z.station_name.notna().all()
        parts.append(z[baseline.columns])
    corrected=pd.concat([baseline[baseline.network_state.ne('NSW')],*parts],ignore_index=True)
    assert not corrected.duplicated(SKEYS).any()
    corrected.to_csv(HERE/'data/regulatory_station_month_2015_2022.csv.gz',index=False)
    comparison=baseline[SKEYS+['observed_pm25','eligible','valid_days']].merge(corrected[SKEYS+['observed_pm25','eligible','valid_days']],on=SKEYS,how='outer',suffixes=('_old','_new'),indicator=True)
    comparison.to_csv(HERE/'audit/station_month_changes.csv.gz',index=False)
    registry=read(PREVIOUS/'audit/station_role_registry.csv')
    d=corrected.merge(registry.drop(columns='country_iso'),on='site_key',how='left',validate='many_to_one')
    assert d.membership_all.notna().all()
    d['base_eligible']=d.eligible&d.eFUA_ID.notna()
    d['selected_all']=d.base_eligible&d.membership_all
    d['selected_role']=d.base_eligible&d.documented_role
    d.to_csv(HERE/'data/station_month_with_roles.csv.gz',index=False)
    old_national=original.load_panel()
    national,national_change=replace_targets(old_national,aggregate(corrected[corrected.eligible&corrected.eFUA_ID.notna()]),'all_networks')
    panels={};summary=[]
    for cohort,select in [('membership_all','selected_all'),('documented_role','selected_role')]:
        old=read(PREVIOUS/f'data/{cohort}_panel.csv.gz')
        panel,changes=replace_targets(old,aggregate(d[d[select]]),cohort)
        panels[cohort]=panel
        for country,c in changes.groupby('country_iso'):
            summary.append({'cohort':cohort,'country_iso':country,'old_rows':int(c._merge.ne('right_only').sum()),
                'new_rows':int(c._merge.ne('left_only').sum()),'changed_target_rows':int(c.target_delta.abs().gt(1e-12).sum()),
                'max_absolute_target_change':float(c.target_delta.abs().max()),
                'changed_station_counts':int(c.station_count_new.ne(c.station_count_old).sum())})
    pd.DataFrame(summary).to_csv(HERE/'audit/panel_impact_summary.csv',index=False)
    dump(HERE/'audit/data_preparation.json',{'created_at':stamp(),'original_station_months':len(baseline),
        'corrected_station_months':len(corrected),'new_station_months':int(comparison._merge.eq('right_only').sum()),
        'original_role_snapshots_preserved':True,'covariates_preserved':True,'national_rows':len(national),
        'panels':summary})
    return old_national,national,panels

def national_models(old,corrected,direction):
    path=BASE/f'results/all_networks/monthly_plus_mixing/{direction}/source_models.joblib'
    archive=joblib.load(path);features=archive['features']
    if direction.startswith('CHN'):return archive
    old_s=old[old.country_iso.eq('AUS')&old.year.le(2018)]
    new_s=corrected[corrected.country_iso.eq('AUS')&corrected.year.le(2018)]
    checks=[];models={}
    for key in ['source']+[f'shuffle_{s}' for s in original.SEEDS]:
        y=old_s[original.Y].to_numpy();new_y=new_s[original.Y].to_numpy()
        if key!='source':
            seed=int(key.split('_')[-1]);y=np.random.default_rng(seed).permutation(y);new_y=np.random.default_rng(seed).permutation(new_y)
        check=clone(archive['models'][key]).fit(old_s[features],y)
        error=float(np.max(np.abs(check.predict(old_s[features])-archive['models'][key].predict(old_s[features]))))
        assert error<1e-10,(key,'national original model reproduction failed',error)
        checks.append({'model':key,'old_refit_max_error':error,'old_source_rows':len(old_s),'new_source_rows':len(new_s)})
        models[key]=clone(archive['models'][key]).fit(new_s[features],new_y)
    result={'models':models,'features':features}
    joblib.dump(result,HERE/'data/corrected_australian_national_models.joblib',compress=3)
    dump(HERE/'audit/national_source_refit_verification.json',checks)
    return result

def main():
    sys.stdout.reconfigure(encoding='utf-8')
    frozen=[]
    for root in [PREVIOUS/'results',PREVIOUS/'data',PREVIOUS/'audit']:
        for p in sorted(root.rglob('*')):
            if p.is_file():frozen.append({'path':str(p),'sha256':sha(p)})
    for p in [PREVIOUS/'PROTOCOL.md',PREVIOUS/'run_regional.py',BASE/'run_monitor_models.py',
              original.FOCUSED/'run_focused.py',BASE/'data/regulatory_station_month_2015_2022.csv.gz']:
        frozen.append({'path':str(p),'sha256':sha(p)})
    dump(HERE/'audit/frozen_input_manifest.json',{'created_at':stamp(),'files':frozen})
    receipt=HERE/'audit/run_receipt.json'
    new={'protocol_sha256':sha(HERE/'PROTOCOL.md'),'script_sha256':sha(__file__),
         'split_sha256':sha(PREVIOUS/'audit/fixed_city_splits.json')}
    if receipt.exists():
        prior=json.loads(receipt.read_text(encoding='utf-8'))
        assert all(prior[k]==v for k,v in new.items()),'Replay script/protocol changed after start'
    else:dump(receipt,{**new,'started_at':stamp()})
    old,national,panels=prepare()
    bundles={direction:national_models(old,national,direction) for direction in ['CHN_to_AUS','AUS_to_CHN']}
    splits=json.loads((PREVIOUS/'audit/fixed_city_splits.json').read_text(encoding='utf-8'))
    original.HERE=HERE
    gate_rows=[];metric_rows=[];locked=[]
    for cohort,d in panels.items():
        for direction,bundle in bundles.items():
            target=direction.split('_to_')[1]
            for seed in original.SEEDS:
                print('START',cohort,direction,seed,flush=True)
                folder=HERE/f'results/{cohort}/{direction}/{seed}';folder.mkdir(parents=True,exist_ok=True)
                if (folder/'predictions.csv.gz').exists():
                    p=read(folder/'predictions.csv.gz')
                else:
                    p,_=original.fit_predict(d,bundle['features'],cohort,direction,seed,folder,bundle,splits[target]['seeds'][str(seed)])
                setting={'scope':cohort,'cohort':cohort,'direction':direction,'seed':seed}
                discovery=p[p.development_city&p.year.eq(2019)]
                validation=p[p.development_city&p.year.eq(2020)]
                for floor in original.FLOORS:
                    lock_path=PREVIOUS/f'results/{cohort}/{direction}/{seed}/lock_{int(floor*100)}.json'
                    lock=json.loads(lock_path.read_text(encoding='utf-8'))
                    updated=dict(lock)
                    if lock['status']=='locked':
                        ds=score(discovery,lock['model'],rulemask(discovery,lock['rule']))
                        vs=score(validation,lock['model'],rulemask(validation,lock['rule']))
                        ok=support(discovery,rulemask(discovery,lock['rule']),floor) and support(validation,rulemask(validation,lock['rule']),floor)
                        updated['discovery_pass']=bool(ok and min(ds['gain'],vs['gain'])>=.05)
                        updated['original_discovery_pass']=lock['discovery_pass']
                        updated['corrected_gain_2019']=ds['gain'];updated['corrected_gain_2020']=vs['gain']
                    updated['original_lock_sha256']=sha(lock_path)
                    dump(folder/f'lock_{int(floor*100)}.json',updated);locked.append(updated)
                    gates,model_metrics=original.audit_lock(p,updated,national[national.country_iso.eq(target)],setting)
                    gate_rows.extend(gates);metric_rows.extend(model_metrics)
                dump(folder/'complete.json',{'completed_at':stamp(),'prediction_sha256':sha(folder/'predictions.csv.gz'),
                     'protocol_sha256':sha(HERE/'PROTOCOL.md'),'replay_script_sha256':sha(__file__),
                     'fit_implementation_sha256':sha(PREVIOUS/'run_regional.py'),'no_condition_reselection':True})
                print('COMPLETE',cohort,direction,seed,flush=True)
    pd.DataFrame(gate_rows).to_csv(HERE/'audit/all_gates.csv',index=False)
    pd.DataFrame(metric_rows).to_csv(HERE/'audit/all_metrics.csv.gz',index=False)
    dump(HERE/'audit/locks.json',locked)
    for f in frozen:assert sha(f['path'])==f['sha256'],f['path']
    dump(HERE/'audit/completion.json',{'completed_at':stamp(),'configurations':12,'locks':len(locked),
        'gate_rows':len(gate_rows),'metric_rows':len(metric_rows),'frozen_files_unchanged':len(frozen),
        'fresh_external_validation':False,'rule_search_performed':False})

if __name__=='__main__':main()
