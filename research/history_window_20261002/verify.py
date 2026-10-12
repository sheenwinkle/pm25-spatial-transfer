"""Verify predictions, label accounting, intersection and reported point metrics."""
import json
import numpy as np
import pandas as pd
from run_windows import HERE,CORRECTED,PREVIOUS,WINDOWS,PRIMARY,old,read,dump,sha,now
from evaluate import stages

def main():
    regression=json.loads((HERE/'audit/12month_regression.json').read_text())
    assert regression['passed'] and regression['max_error']<1e-8
    receipt=json.loads((HERE/'audit/run_receipt.json').read_text())
    assert receipt['protocol_sha256']==sha(HERE/'PROTOCOL.md') and receipt['script_sha256']==sha(HERE/'run_windows.py')
    for f in receipt['inputs']:assert sha(f['path'])==f['sha256']
    splits=json.loads((PREVIOUS/'audit/fixed_city_splits.json').read_text())
    metric=read(HERE/'audit/metrics.csv.gz');primary=read(HERE/'audit/primary.csv')
    budget=read(HERE/'audit/label_accounting.csv');available=read(HERE/'audit/availability.csv')
    count=rolling=newfits=reused=0;max_error=0.0
    for cohort in old.COHORTS:
        panel=read(CORRECTED/f'data/{cohort}_panel.csv.gz')
        for direction in ['CHN_to_AUS','AUS_to_CHN']:
            target=direction.split('_to_')[1];target_panel=panel[panel.country_iso.eq(target)]
            for seed in old.SEEDS:
                ps={w:read(HERE/f'window_{w}/results/{cohort}/{direction}/{seed}/predictions.csv.gz') for w in WINDOWS}
                common=set.intersection(*(set(zip(p.eFUA_ID,p.t)) for p in ps.values()))
                for w,p in ps.items():
                    folder=HERE/f'window_{w}/results/{cohort}/{direction}/{seed}'
                    done=json.loads((folder/'complete.json').read_text())
                    assert done['prediction_sha256']==sha(folder/'predictions.csv.gz')
                    assert np.isfinite(p[old.MODELS]).all(axis=None)
                    assert not p.duplicated(['eFUA_ID','t']).any()
                    assert set(zip(p.eFUA_ID,p.t))<=set(zip(ps[12].eFUA_ID,ps[12].t))
                    if done['mode']=='new_fit':newfits+=1
                    else:reused+=1
                    ids=splits[target]['seeds'][str(seed)]
                    assert p.development_city.eq(p.eFUA_ID.isin(ids)).all()
                    audit=read(folder/'fit_audit.csv');rolling+=len(audit)
                    actual_labels=set()
                    for a in audit.itertuples():
                        fit=target_panel[target_panel.eFUA_ID.isin(ids)&target_panel.t.between(a.prediction_t-w,a.prediction_t-1)]
                        assert len(fit)>=5 and len(fit)==a.labels
                        assert a.latest_fit_t==fit.t.max()<a.prediction_t
                        assert a.earliest_fit_t==fit.t.min()>=a.prediction_t-w
                        assert set(json.loads(a.fit_cities))==set(fit.eFUA_ID.unique())
                        actual_labels.update(zip(fit.eFUA_ID,fit.t))
                        assert p[p.t.eq(a.prediction_t)].fit_labels.eq(a.labels).all()
                    labels=read(folder/'calibration_rows.csv.gz')
                    assert actual_labels==set(zip(labels.eFUA_ID,labels.t))
                    b=budget[budget.cohort.eq(cohort)&budget.direction.eq(direction)&budget.seed.eq(seed)&budget.window.eq(w)].iloc[0]
                    assert b.unique_labels_across_run==len(actual_labels)
                    assert b.labels_per_fit_min==audit.labels.min() and b.labels_per_fit_max==audit.labels.max()
                    assert b.labels_per_fit_median==audit.labels.median() and b.skipped_prediction_months==48-len(audit)
                    common_mask=np.array([(city,t) in common for city,t in zip(p.eFUA_ID,p.t)])
                    for stage,use in stages(p).items():
                        for selection,frame in [('native',p[use]),('common',p[use&common_mask])]:
                            filt=lambda x:x[x.cohort.eq(cohort)&x.direction.eq(direction)&x.seed.eq(seed)&x.window.eq(w)&x.stage.eq(stage)&x.selection.eq(selection)]
                            rows=filt(metric).set_index('model')
                            assert set(rows.index)==set(old.MODELS)
                            yy=frame.observed_pm25.to_numpy();denom=np.sum((yy-yy.mean())**2) if len(yy) else 0
                            errors={}
                            for model in old.MODELS:
                                ee=frame[model].to_numpy()-yy
                                expected={'n':len(frame),'fuas':frame.eFUA_ID.nunique(),'months':frame.t.nunique(),
                                    'rmse':np.sqrt(np.mean(ee**2)) if len(ee) else np.nan,
                                    'mae':np.mean(np.abs(ee)) if len(ee) else np.nan,
                                    'bias':np.mean(ee) if len(ee) else np.nan,
                                    'r2':1-np.sum(ee**2)/denom if denom>0 else np.nan}
                                errors[model]=expected['rmse']
                                for key,value in expected.items():
                                    reported=rows.loc[model,key]
                                    assert np.isclose(reported,value,atol=1e-9,rtol=0,equal_nan=True),(model,stage,key)
                                    if np.isfinite(value):max_error=max(max_error,abs(value-reported))
                                count+=1
                            pr=filt(primary).iloc[0]
                            denom_frame=ps[12][stages(ps[12])[stage]]
                            coverage=len(frame)/len(denom_frame) if len(denom_frame) else 0
                            assert np.isclose(pr.coverage,coverage)
                            gain=1-errors[PRIMARY]/max(min(errors[k] for k in old.REFS),1e-12) if len(frame) else np.nan
                            assert np.isclose(pr.gain,gain,atol=1e-10,equal_nan=True)
                            av=filt(available).iloc[0]
                            assert av.rows==len(frame) and av.full_12month_stage_rows==len(denom_frame)
                            support=coverage>=.5 and set(frame.year.unique())==set(denom_frame.year.unique())
                            for year in denom_frame.year.unique():
                                z=frame[frame.year.eq(year)]
                                support &= len(z)>=24 and z.eFUA_ID.nunique()>=3 and z.t.nunique()>=6
                            if 'audit' in stage:support &= denom_frame.year.nunique()>=2
                            assert pr.support_pass==support
                            lows=pr[[k for k in pr.index if k.endswith('_low')]]
                            boot='audit' in stage and lows.notna().all() and lows.gt(0).all()
                            assert pr.bootstrap_pass==boot
                            assert pr.screen_pass==bool(support and boot and gain>=.05)
    assert count==len(metric)==12960 and newfits==37 and reused==11
    j=read(HERE/'audit/joint_screens.csv')
    for r in j.itertuples():
        z=primary[primary.cohort.eq(r.cohort)&primary.direction.eq(r.direction)&primary.seed.eq(r.seed)&primary.window.eq(r.window)&primary.selection.eq('common')&primary.stage.isin(['temporal_audit','new_city_audit'])]
        assert len(z)==2 and r.joint_pass==bool(z.screen_pass.all())
    dump(HERE/'audit/VERIFICATION.json',{'verified_at':now(),'passed':True,'configurations':48,
        'new_fitted_configurations':newfits,'reused_configurations':reused,'rolling_windows_checked':rolling,
        'metric_rows_recomputed':count,'max_metric_error':max_error,'12month_regression_max_error':regression['max_error'],
        'original_input_hashes_verified':len(receipt['inputs']),'primary_3month_joint_passes':int(j[j.window.eq(3)&j.cohort.eq('membership_all')].joint_pass.sum()),
        'all_window_joint_passes':int(j.joint_pass.sum()),'fresh_external_validation':False,
        'global_label_budget_reduction_demonstrated':False})
    print(j.groupby(['cohort','direction','window']).joint_pass.agg(['sum','count']).to_string(),flush=True)
    print('Verified',count,'metrics;',rolling,'rolling windows',flush=True)

if __name__=='__main__':main()
