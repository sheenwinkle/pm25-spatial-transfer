"""Independent saved-output checks and descriptive, non-selection diagnostics."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from common import HERE, DATA, AUDIT, RESULTS, MONTHLY, dump, digest

KEYS=['cohort','feature_set','direction','seed','window_months']

def read_predictions(path):
    # Inclusive thresholds need exact binary-float recovery, not an approximate parser.
    return pd.read_csv(path,float_precision='round_trip')

def main():
    folders=sorted(p.parent for p in RESULTS.glob('*/*/*/manifest.json'))
    assert len(folders)==10, f'Expected six primary and four supplemental branches, got {len(folders)}'
    allmetrics=[];allgates=[];allbudgets=[];verification=[];stable=[];candidates=[]
    for folder in folders:
        cohort,fs,direction=folder.relative_to(RESULTS).parts
        metrics=pd.read_csv(folder/'metrics.csv');gates=pd.read_csv(folder/'gates.csv')
        allmetrics.append(metrics);allgates.append(gates);allbudgets.append(pd.read_csv(folder/'budgets.csv'))
        manifest=json.loads((folder/'manifest.json').read_text());audit_first=manifest.get('audit_first',2021)
        splits=pd.read_csv(folder/'city_splits.csv')
        for seed in [5993,5994,5995]:
            ids=set(splits.loc[splits.seed.eq(seed)&splits.development,'eFUA_ID'])
            for w in [3,6,12]:
                path=folder/f'predictions_{seed}_{w}.csv.gz';d=read_predictions(path)
                lock=json.loads((folder/f'lock_{seed}_{w}.json').read_text())
                assert not d.duplicated(['eFUA_ID','year','month']).any()
                assert (d.development_city==d.eFUA_ID.isin(ids)).all()
                accepted=np.ones(len(d),bool)
                for a in lock['rule']:
                    accepted &= d[a['feature']].le(a['threshold']) if a['op']=='<=' else d[a['feature']].ge(a['threshold'])
                    stable.append({'cohort':cohort,'feature_set':fs,'direction':direction,'seed':seed,'window_months':w,
                        'feature':a['feature'],'operator':a['op'],'threshold':a['threshold'],'discovery_pass':lock['discovery_pass']})
                assert np.array_equal(accepted,d.accepted.to_numpy())
                assert np.allclose(d.selected_prediction,d[lock['model']],atol=1e-10,rtol=0)
                fit=pd.read_csv(folder/f'fit_audit_{seed}_{w}.csv');assert (fit.latest_fit_t<fit.month_t).all()
                assert (fit.latest_fit_t>=fit.month_t-w).all() and (fit.labels<=len(ids)*w).all()
                stages={'discovery':d.development_city&d.year.lt(audit_first),
                    'temporal_audit':d.development_city&d.year.ge(audit_first),
                    'new_city_audit':~d.development_city&d.year.ge(audit_first),
                    'new_weather_grid_audit':~d.development_city&d.new_weather_grid&d.year.ge(audit_first)}
                m=metrics[metrics.seed.eq(seed)&metrics.window_months.eq(w)]
                maxerr=0.0
                for _,r in m.iterrows():
                    z=d[stages[r.stage]]
                    if r.selection=='accepted':z=z[z.accepted]
                    assert len(z)==r.n and z.eFUA_ID.nunique()==r.fuas and z.t.nunique()==r.months
                    if not len(z):continue
                    y=z.observed_pm25.to_numpy();p=z[r.model].to_numpy()
                    calculated={'rmse':np.sqrt(mean_squared_error(y,p)),'mae':mean_absolute_error(y,p),'r2':r2_score(y,p),'bias':np.mean(p-y)}
                    for name,v in calculated.items():
                        err=abs(v-r[name]);maxerr=max(maxerr,err);assert err<1e-7,(path,name,err)
                g=gates[gates.seed.eq(seed)&gates.window_months.eq(w)]
                for _,r in g.iterrows():
                    full=d[stages[r.stage]];z=full[full.accepted]
                    assert abs(r.coverage-len(z)/len(full))<1e-10
                    locals_=['target_mean','target_ridge','target_rbf','target_rf','target_extra','target_lgbm']
                    neg=[f'shuffled_kernel_{s}' for s in [5993,5994,5995]]
                    rmse=lambda c:np.sqrt(np.mean((z[c]-z.observed_pm25)**2))
                    improvement=1-rmse(lock['model'])/min(rmse(c) for c in locals_+neg)
                    assert abs(improvement-r.relative_gain)<1e-10
                verification.append({'cohort':cohort,'feature_set':fs,'direction':direction,'seed':seed,'window_months':w,
                    'prediction_sha256':digest(path),'metric_records_checked':len(m),'max_metric_abs_difference':maxerr,
                    'accepted_rows_checked':len(d),'chronological_fits_checked':len(fit),'pass':True})
                candidates.append({'cohort':cohort,'feature_set':fs,'direction':direction,'seed':seed,'window_months':w,
                    'discovery_pass':lock['discovery_pass'],'model':lock['model'],'rule':json.dumps(lock['rule']),
                    'complexity':len(lock['rule'])})
    m=pd.concat(allmetrics,ignore_index=True);g=pd.concat(allgates,ignore_index=True);b=pd.concat(allbudgets,ignore_index=True)
    for filename,data in [('all_metrics',m),('all_gates',g),('all_budgets',b),('rule_endpoints',pd.DataFrame(stable)),
                          ('frozen_candidates',pd.DataFrame(candidates)),('saved_prediction_verification',pd.DataFrame(verification))]:
        data.to_csv(AUDIT/f'{filename}.csv',index=False)
    pair=g[g.stage.isin(['temporal_audit','new_city_audit'])].groupby(KEYS).agg(stages=('stage','nunique'),passed=('pass','all')).reset_index()
    pair['both_audits_pass']=pair.stages.eq(2)&pair.passed;pair.to_csv(AUDIT/'replication.csv',index=False)
    counts=pair.groupby(['cohort','feature_set','direction']).agg(configurations=('seed','size'),passes=('both_audits_pass','sum')).reset_index()
    counts.to_csv(AUDIT/'replication_counts.csv',index=False)
    d=pd.read_csv(DATA/'all_networks_model_panel.csv.gz')
    station=pd.read_csv(DATA/'regulatory_station_month_2015_2022.csv.gz')
    station.groupby(['country_iso','network_state']).agg(rows=('site_key','size'),stations=('site_key','nunique'),
        eligible=('eligible','sum'),polygon_matched=('eFUA_ID','count')).to_csv(AUDIT/'station_support_by_network.csv')
    changes=[]
    for (country,city),z in d.groupby(['country_iso','eFUA_ID']):
        z=z.sort_values(['year','month']);sets=[set(str(s).split('|')) for s in z.sites]
        for i in range(1,len(z)):
            if z.iloc[i].t-z.iloc[i-1].t!=1:continue
            changes.append({'country_iso':country,'eFUA_ID':city,'year':int(z.iloc[i].year),'month':int(z.iloc[i].month),
                'jaccard':len(sets[i]&sets[i-1])/len(sets[i]|sets[i-1]),'same_composition':sets[i]==sets[i-1]})
    pd.DataFrame(changes).to_csv(AUDIT/'station_composition_changes.csv.gz',index=False)
    # The +10-hour alternative is an archive-day sensitivity, with annual-file edge limitations.
    q=pd.read_csv(AUDIT/'qld_plus10_hour_monthly.csv.gz');q=q[q.eligible&q.year.between(2015,2022)]
    p=station[station.network_state.eq('QLD')&station.eligible]
    z=p.merge(q[['site_key','year','month','observed_pm25','valid_days']],on=['site_key','year','month'],suffixes=('_native','_plus10'),validate='one_to_one')
    z['difference']=z.observed_pm25_plus10-z.observed_pm25_native
    z.to_csv(AUDIT/'qld_time_basis_sensitivity.csv',index=False)
    dump(AUDIT/'quality_sensitivity_summary.json',{'qld_overlapping_station_months':len(z),
        'qld_mean_abs_difference':z.difference.abs().mean(),'qld_max_abs_difference':z.difference.abs().max(),
        'qld_note':'Descriptive time-basis perturbation, not a verified correction; annual-file edges may lose a cross-year day.',
        'station_composition_consecutive_pairs':len(changes),'station_composition_unchanged_fraction':pd.DataFrame(changes).groupby('country_iso').same_composition.mean().to_dict()})
    dump(AUDIT/'verification_summary.json',{'status':'passed','configurations':len(verification),
        'metric_records_recomputed':sum(r['metric_records_checked'] for r in verification),
        'max_metric_abs_difference':max(r['max_metric_abs_difference'] for r in verification),
        'frozen_rule_rows_checked':sum(r['accepted_rows_checked'] for r in verification),
        'chronological_fits_checked':sum(r['chronological_fits_checked'] for r in verification),
        'both_audits_pass':int(pair.both_audits_pass.sum()),
        'scope':'Saved predictions, exact rules, split identities and fitting-time bounds checked; model internals not independently reimplemented.'})
    print('ALL_OUTPUTS_VERIFIED',len(verification),flush=True)

if __name__=='__main__':main()
