"""Independent metric, temporal-order, frozen-rule and data-version checks."""
import json
import importlib.metadata
import numpy as np
import pandas as pd
from replay import HERE, PREVIOUS, BASE, READINESS, read, dump, sha, stamp, KEYS, SKEYS, original

def calc(frame,prediction):
    y=frame.observed_pm25.to_numpy(float);p=frame[prediction].to_numpy(float)
    if not len(y):return {'n':0,'fuas':0,'months':0,'rmse':np.nan,'mae':np.nan,'bias':np.nan,'r2':np.nan}
    error=p-y;denom=float(np.dot(y-y.mean(),y-y.mean()))
    return {'n':len(y),'fuas':frame.eFUA_ID.nunique(),'months':frame.t.nunique(),
            'rmse':float(np.sqrt(np.dot(error,error)/len(y))),'mae':float(np.abs(error).sum()/len(y)),
            'bias':float(error.sum()/len(y)),'r2':1-float(np.dot(error,error))/denom if denom>0 else np.nan}

def mask(frame,rule):
    result=np.ones(len(frame),dtype=bool)
    for atom in rule:
        v=frame[atom['feature']].to_numpy(float)
        if atom['op']=='<=':result &= np.isfinite(v)&(v<=atom['threshold'])
        elif atom['op']=='>=':result &= np.isfinite(v)&(v>=atom['threshold'])
        else:raise AssertionError(atom['op'])
    return result

def main():
    completion=json.loads((HERE/'audit/completion.json').read_text(encoding='utf-8'))
    assert completion['configurations']==12 and completion['locks']==48
    frozen=json.loads((HERE/'audit/frozen_input_manifest.json').read_text(encoding='utf-8'))
    for f in frozen['files']:assert sha(f['path'])==f['sha256'],f['path']
    receipt=json.loads((HERE/'audit/run_receipt.json').read_text(encoding='utf-8'))
    assert receipt['script_sha256']==sha(HERE/'replay.py')
    assert receipt['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert receipt['split_sha256']==sha(PREVIOUS/'audit/fixed_city_splits.json')
    boundary=json.loads((READINESS/'audit/nsw_boundary_repair_summary.json').read_text(encoding='utf-8'))
    for f in boundary['source_files']:
        assert sha(f['original_path'])==f['original_sha256']
        assert sha(f['patch_path'])==f['patch_sha256']
    old=read(BASE/'data/regulatory_station_month_2015_2022.csv.gz')
    new=read(HERE/'data/regulatory_station_month_2015_2022.csv.gz')
    assert len(new)==len(old) and not new.duplicated(SKEYS).any()
    pd.testing.assert_frame_equal(old[old.network_state.ne('NSW')].reset_index(drop=True),new[new.network_state.ne('NSW')].reset_index(drop=True))
    assert new.eligible.eq(new.valid_days>=new.required_days).all()
    # Cross-check the independently rebuilt old-period table against the prior
    # branch's append-and-recalculate correction candidate, including eligibility.
    candidate=read(READINESS/'data/regulatory_station_month_existing_rows_boundary_corrected_2015_2022.csv.gz')
    check=new.merge(candidate,on=SKEYS,validate='one_to_one',suffixes=('_a','_b'))
    assert np.allclose(check.observed_pm25_a,check.observed_pm25_b,atol=1e-12,rtol=0)
    assert check.valid_days_a.eq(check.valid_days_b).all() and check.eligible_a.eq(check.eligible_b).all()
    splits=json.loads((PREVIOUS/'audit/fixed_city_splits.json').read_text(encoding='utf-8'))
    measured=read(HERE/'audit/all_metrics.csv.gz')
    gates=read(HERE/'audit/all_gates.csv')
    verified=[];rolling=0;predictions=0;lock_count=0
    max_error=0.0
    for cohort in original.COHORTS:
        panel=read(HERE/f'data/{cohort}_panel.csv.gz')
        for direction in ['CHN_to_AUS','AUS_to_CHN']:
            target=direction.split('_to_')[1];target_panel=panel[panel.country_iso.eq(target)]
            for seed in original.SEEDS:
                folder=HERE/f'results/{cohort}/{direction}/{seed}'
                p=read(folder/'predictions.csv.gz');predictions+=len(p)
                done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                assert done['prediction_sha256']==sha(folder/'predictions.csv.gz')
                assert done['fit_implementation_sha256']==sha(PREVIOUS/'run_regional.py')
                assert np.isfinite(p[original.MODELS]).all(axis=None)
                assert not p.duplicated(['eFUA_ID','t']).any()
                ids=splits[target]['seeds'][str(seed)]
                assert p.development_city.eq(p.eFUA_ID.isin(ids)).all()
                fit=read(folder/'fit_audit.csv');rolling+=len(fit)
                for row in fit.itertuples():
                    training=target_panel[target_panel.eFUA_ID.isin(ids)&target_panel.t.between(row.prediction_t-12,row.prediction_t-1)]
                    assert row.latest_fit_t<row.prediction_t and row.earliest_fit_t>=row.prediction_t-12
                    assert len(training)==row.labels
                    assert p[p.t.eq(row.prediction_t)].fit_labels.eq(row.labels).all()
                    assert set(json.loads(row.fit_cities))<=set(ids)
                stages={'discovery_2019':p.development_city&p.year.eq(2019),
                        'validation_2020':p.development_city&p.year.eq(2020),
                        'temporal_audit':p.development_city&p.year.ge(2021),
                        'new_city_audit':~p.development_city&p.year.ge(2021),
                        'new_grid_audit':~p.development_city&p.new_weather_grid&p.year.ge(2021)}
                for floor in original.FLOORS:
                    file=f'lock_{int(floor*100)}.json'
                    lock=json.loads((folder/file).read_text(encoding='utf-8'))
                    old_lock=json.loads((PREVIOUS/f'results/{cohort}/{direction}/{seed}'/file).read_text(encoding='utf-8'))
                    for k in ['status','model','rule','floor']:
                        assert lock.get(k)==old_lock.get(k),(cohort,direction,seed,k)
                    assert lock['original_lock_sha256']==sha(PREVIOUS/f'results/{cohort}/{direction}/{seed}'/file)
                    lock_count+=1
                    for stage,use in stages.items():
                        full=p[use];accepted=full[mask(full,lock['rule'])]
                        g=gates[gates.cohort.eq(cohort)&gates.direction.eq(direction)&gates.seed.eq(seed)&gates.floor.eq(floor)&gates.stage.eq(stage)].iloc[0]
                        refs={m:calc(accepted,m)['rmse'] for m in original.REFS}
                        rms=calc(accepted,lock['model'])['rmse']
                        gain=1-rms/min(refs.values()) if len(accepted) else np.nan
                        assert np.isclose(g.gain,gain,atol=1e-10,equal_nan=True)
                        assert np.isclose(g.coverage,len(accepted)/len(full) if len(full) else 0)
                        for selection,frame in [('full_scope',full),('accepted',accepted)]:
                            subset=measured[measured.cohort.eq(cohort)&measured.direction.eq(direction)&measured.seed.eq(seed)&measured.floor.eq(floor)&measured.stage.eq(stage)&measured.selection.eq(selection)].set_index('model')
                            assert set(subset.index)==set(original.MODELS)
                            for model in original.MODELS:
                                value=calc(frame,model);reported=subset.loc[model]
                                for key,expected in value.items():
                                    actual=reported[key]
                                    assert np.isclose(actual,expected,atol=1e-8,rtol=0,equal_nan=True),(model,stage,key,actual,expected)
                                    if np.isfinite(actual) and np.isfinite(expected):max_error=max(max_error,abs(float(actual)-expected))
                                verified.append((cohort,direction,seed,floor,stage,selection,model))
    assert len(verified)==len(measured)==12960 and rolling==576 and lock_count==48
    before=read(PREVIOUS/'audit/all_gates.csv')
    keys=['cohort','direction','seed','floor','stage']
    paired=before.merge(gates,on=keys,validate='one_to_one',suffixes=('_before','_after'))
    assert len(paired)==len(gates)==240
    assert paired.model_before.eq(paired.model_after).all() and paired.rule_before.eq(paired.rule_after).all()
    for metric in ['gain','rmse','mae','r2','coverage','n','fuas','months']:
        paired['delta_'+metric]=paired[metric+'_after']-paired[metric+'_before']
    paired.to_csv(HERE/'audit/paired_gate_changes.csv',index=False)
    settings=['cohort','direction','seed','floor']
    joint=[]
    for key,g in paired.groupby(settings):
        z=g[g.stage.isin(['temporal_audit','new_city_audit'])]
        assert len(z)==2
        joint.append(dict(zip(settings,key))|{'joint_before':bool(z.screen_pass_before.all()),
            'joint_after':bool(z.screen_pass_after.all()),'early_before':bool(z.discovery_pass_before.iloc[0]),
            'early_after':bool(z.discovery_pass_after.iloc[0])})
    j=pd.DataFrame(joint);j.to_csv(HERE/'audit/paired_joint_outcomes.csv',index=False)
    summary=[]
    for (cohort,direction),g in paired[paired.stage.eq('new_city_audit')].groupby(['cohort','direction']):
        jj=j[j.cohort.eq(cohort)&j.direction.eq(direction)]
        summary.append({'cohort':cohort,'direction':direction,'settings':len(g),
            'new_city_gain_min_before':g.gain_before.min(),'new_city_gain_max_before':g.gain_before.max(),
            'new_city_gain_min_after':g.gain_after.min(),'new_city_gain_max_after':g.gain_after.max(),
            'new_city_positive_gain_after':int(g.gain_after.gt(0).sum()),
            'max_absolute_gain_change':g.delta_gain.abs().max(),
            'joint_passes_before':int(jj.joint_before.sum()),'joint_passes_after':int(jj.joint_after.sum())})
    pd.DataFrame(summary).to_csv(HERE/'audit/paired_direction_summary.csv',index=False)
    dump(HERE/'audit/VERIFICATION.json',{'verified_at':stamp(),'passed':True,'frozen_files_verified':len(frozen['files']),
        'raw_input_and_patch_hashes_checked':len(boundary['source_files'])*2,
        'configurations':12,'rolling_fits':rolling,'fixed_condition_settings':lock_count,'metrics_recomputed':len(verified),
        'prediction_rows_across_configurations':predictions,'max_metric_error':max_error,
        'joint_passes_before':int(j.joint_before.sum()),'joint_passes_after':int(j.joint_after.sum()),
        'early_passes_before':int(j.early_before.sum()),'early_passes_after':int(j.early_after.sum()),
        'all_new_city_positive_gains_after':int(paired[paired.stage.eq('new_city_audit')].gain_after.gt(0).sum()),
        'new_condition_search':False,'fresh_external_test':False,
        'versions':{n:importlib.metadata.version(n) for n in ['pandas','numpy','scikit-learn','lightgbm','scipy']}})
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    print('Verified',len(verified),'metrics;',rolling,'rolling fits;',lock_count,'unchanged locks',flush=True)

if __name__=='__main__':main()
