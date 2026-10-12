"""Independent label selection and point-metric verification."""
import json
import numpy as np
import pandas as pd
from run_gap_models import HERE,WINDOWS,REGIONAL,CORRECTED,old,read,dump,sha,now
from evaluate import PRIMARY,folder_for,stages


def main():
    receipt=json.loads((HERE/'audit/model_run_receipt.json').read_text(encoding='utf-8'))
    assert receipt['protocol_sha256']==sha(HERE/'PROTOCOL.md')
    assert receipt['script_sha256']==sha(HERE/'run_gap_models.py')
    for item in receipt['inputs']:assert sha(item['path'])==item['sha256']
    regression=json.loads((HERE/'audit/fixed_path_regression.json').read_text())
    assert regression['passed'] and regression['maximum_prediction_error']<1e-8
    splits=json.loads((REGIONAL/'audit/fixed_city_splits.json').read_text())
    metrics=read(HERE/'audit/metrics.csv.gz');primary=read(HERE/'audit/primary.csv')
    budgets=read(HERE/'audit/label_accounting.csv');available=read(HERE/'audit/availability.csv')
    count=rolling=gap_rolling=0;maximum=0.;gap_configurations=0;prediction_labels=0
    for cohort in old.COHORTS:
        panel=read(CORRECTED/f'data/{cohort}_panel.csv.gz')
        for direction in ['CHN_to_AUS','AUS_to_CHN']:
            target=direction.split('_to_')[1];target_panel=panel[panel.country_iso.eq(target)]
            for seed in old.SEEDS:
                ids=splits[target]['seeds'][str(seed)]
                full=read(WINDOWS/f'window_12/results/{cohort}/{direction}/{seed}/predictions.csv.gz')
                for width in [3,1]:
                    ps={m:read(folder_for(cohort,direction,seed,width,m)/'predictions.csv.gz') for m in ['fixed','valid']}
                    keys={m:set(zip(p.eFUA_ID,p.t)) for m,p in ps.items()}
                    common=keys['fixed']&keys['valid'];added=keys['valid']-keys['fixed']
                    assert keys['fixed']<=keys['valid']<=set(zip(full.eFUA_ID,full.t))
                    for method,p in ps.items():
                        folder=folder_for(cohort,direction,seed,width,method)
                        done=json.loads((folder/'complete.json').read_text(encoding='utf-8'))
                        assert done['prediction_sha256']==sha(folder/'predictions.csv.gz')
                        assert np.isfinite(p[old.MODELS]).all(axis=None)
                        assert not p.duplicated(['eFUA_ID','t']).any()
                        assert p.development_city.eq(p.eFUA_ID.isin(ids)).all()
                        truth=p.merge(target_panel[['eFUA_ID','t','observed_pm25']],on=['eFUA_ID','t'],validate='one_to_one',suffixes=('','_original'))
                        assert np.array_equal(truth.observed_pm25,truth.observed_pm25_original)
                        prediction_labels+=len(p)
                        if method=='valid':
                            gap_configurations+=1
                            assert done['script_sha256']==receipt['script_sha256'] and done['protocol_sha256']==receipt['protocol_sha256']
                            selected=read(folder/'selected_label_rows.csv.gz');history=read(folder/'history_audit.csv')
                            assert len(history)==48
                        audit=read(folder/'fit_audit.csv');rolling+=len(audit)
                        if method=='valid':gap_rolling+=len(audit)
                        label_set=set()
                        # Independent per-city sorting, not the production groupby/tail selector.
                        for month in sorted(target_panel[target_panel.year.ge(2019)].t.unique()):
                            candidates=target_panel[target_panel.eFUA_ID.isin(ids)&target_panel.t.between(month-(12 if method=='valid' else width),month-1)]
                            if method=='valid':
                                chosen=[g.sort_values('t',ascending=False).iloc[:width] for _,g in candidates.groupby('eFUA_ID')]
                                chosen=pd.concat(chosen) if chosen else candidates.iloc[:0]
                            else:chosen=candidates
                            expected=set(zip(chosen.eFUA_ID,chosen.t))
                            if method=='valid':
                                z=selected[selected.prediction_t.eq(month)]
                                assert len(z)==len(chosen) and set(zip(z.eFUA_ID,z.t))==expected
                                assert (z.label_age.eq(month-z.t)&z.label_age.between(1,12)).all()
                                assert z.groupby('eFUA_ID').size().le(width).all()
                                assert z.fit_performed.eq(len(chosen)>=5).all()
                                h=history[history.prediction_t.eq(month)].iloc[0]
                                assert h.selected_rows==len(chosen) and h.fit_performed==(len(chosen)>=5)
                                assert h.labels_older_than_nominal_window==int((month-chosen.t).gt(width).sum())
                            fitted=audit[audit.prediction_t.eq(month)]
                            if len(chosen)<5:assert fitted.empty and not p.t.eq(month).any();continue
                            assert len(fitted)==1
                            a=fitted.iloc[0]
                            assert a.labels==len(chosen) and a.latest_fit_t==chosen.t.max()<month
                            assert a.earliest_fit_t==chosen.t.min()>=month-(12 if method=='valid' else width)
                            assert set(json.loads(a.fit_cities))==set(chosen.eFUA_ID)
                            assert p[p.t.eq(month)].fit_labels.eq(len(chosen)).all()
                            label_set.update(expected)
                        used=read(folder/'calibration_rows.csv.gz')
                        assert len(used)==len(label_set) and set(zip(used.eFUA_ID,used.t))==label_set
                        filt=lambda x:x[x.cohort.eq(cohort)&x.direction.eq(direction)&x.seed.eq(seed)&x.width.eq(width)&x.method.eq(method)]
                        budget=filt(budgets).iloc[0]
                        assert budget.fits==len(audit) and budget.skipped_prediction_months==48-len(audit)
                        assert budget.unique_labels_across_run==len(used) and budget.predicted_rows==len(p)
                        assert budget.labels_per_fit_median==audit.labels.median()
                        masks={'native':np.ones(len(p),bool),'common':np.array([(c,t) in common for c,t in zip(p.eFUA_ID,p.t)])}
                        if method=='valid':masks['added']=np.array([(c,t) in added for c,t in zip(p.eFUA_ID,p.t)])
                        for stage,use in stages(p).items():
                            denom=full[stages(full)[stage]]
                            for selection,mask in masks.items():
                                frame=p[use&mask];y=frame.observed_pm25.to_numpy();v=np.sum((y-y.mean())**2) if len(y) else 0
                                f=lambda x:filt(x)[filt(x).stage.eq(stage)&filt(x).selection.eq(selection)]
                                mr=f(metrics).set_index('model');errors={}
                                for model in old.MODELS:
                                    e=frame[model].to_numpy()-y
                                    want={'n':len(frame),'fuas':frame.eFUA_ID.nunique(),'months':frame.t.nunique(),
                                        'rmse':np.sqrt(np.mean(e**2)) if len(e) else np.nan,
                                        'mae':np.mean(np.abs(e)) if len(e) else np.nan,'bias':e.mean() if len(e) else np.nan,
                                        'r2':1-np.sum(e**2)/v if v>0 else np.nan}
                                    errors[model]=want['rmse']
                                    for key,value in want.items():
                                        reported=mr.loc[model,key]
                                        assert np.isclose(reported,value,atol=1e-9,rtol=0,equal_nan=True),(model,stage,key)
                                        if np.isfinite(value):maximum=max(maximum,abs(reported-value))
                                    count+=1
                                pr=f(primary).iloc[0];av=f(available).iloc[0]
                                coverage=len(frame)/len(denom) if len(denom) else 0
                                assert av.rows==len(frame) and av.eligible_stage_rows==len(denom)
                                assert np.isclose(pr.coverage,coverage)
                                gain=1-errors[PRIMARY]/max(min(errors[k] for k in old.REFS),1e-12) if len(y) else np.nan
                                assert np.isclose(gain,pr.gain,atol=1e-10,equal_nan=True)
                                support=coverage>=.5 and set(frame.year.unique())==set(denom.year.unique())
                                for year in denom.year.unique():
                                    z=frame[frame.year.eq(year)]
                                    support &= len(z)>=24 and z.eFUA_ID.nunique()>=3 and z.t.nunique()>=6
                                if 'audit' in stage and selection!='added':support &= denom.year.nunique()>=2
                                assert support==pr.support_pass
                                lows=pr[[c for c in pr.index if c.endswith('_low')]]
                                boot='audit' in stage and selection!='added' and lows.notna().all() and lows.gt(0).all()
                                assert pr.bootstrap_pass==boot
                                assert pr.screen_pass==bool(selection!='added' and support and boot and gain>=.05)
    assert gap_configurations==24 and count==len(metrics)==16200
    paired=read(HERE/'audit/paired_models.csv.gz')
    for r in paired.itertuples():
        z=metrics[metrics.cohort.eq(r.cohort)&metrics.direction.eq(r.direction)&metrics.seed.eq(r.seed)&metrics.width.eq(r.width)&metrics.stage.eq(r.stage)&metrics.selection.eq('common')&metrics.model.eq(r.model)].set_index('method')
        assert np.isclose(r.delta_rmse_valid_minus_fixed,z.loc['valid','rmse']-z.loc['fixed','rmse'],atol=1e-9,equal_nan=True)
    joint=read(HERE/'audit/joint_screens.csv')
    for r in joint.itertuples():
        z=primary[primary.cohort.eq(r.cohort)&primary.direction.eq(r.direction)&primary.seed.eq(r.seed)&primary.width.eq(r.width)&primary.method.eq(r.method)&primary.selection.eq('common')&primary.stage.isin(['temporal_audit','new_city_audit'])]
        assert len(z)==2 and r.joint_pass==z.screen_pass.all()
    raw=json.loads((HERE/'audit/raw_audit_receipt.json').read_text(encoding='utf-8'))
    assert raw['maximum_mean_error']<1e-10 and raw['archive_md5_verified']
    dump(HERE/'audit/VERIFICATION.json',{'verified_at':now(),'passed':True,'new_configurations':gap_configurations,
        'rolling_fits_checked':rolling,'new_gap_rolling_fits_checked':gap_rolling,'metric_rows_recomputed':count,
        'max_metric_error':maximum,'prediction_label_rows_equal_to_input':prediction_labels,
        'paired_model_rows_checked':len(paired),'original_input_hashes_verified':len(receipt['inputs']),
        'fixed_path_regression_max_error':regression['maximum_prediction_error'],
        'primary_valid3_joint_passes':int(joint[joint.method.eq('valid')&joint.width.eq(3)&joint.cohort.eq('membership_all')].joint_pass.sum()),
        'all_valid_joint_passes':int(joint[joint.method.eq('valid')].joint_pass.sum()),
        'fresh_external_validation':False,'synthetic_targets_used':False})
    print('VERIFIED',count,'metric rows;',rolling,'rolling fits',flush=True)


if __name__=='__main__':main()
