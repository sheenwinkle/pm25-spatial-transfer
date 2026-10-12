"""Paired calendar/observed-history evaluation; no target filtering or search."""
import json
import numpy as np
import pandas as pd
from run_gap_models import HERE,WINDOWS,CORRECTED,old,read,dump,now
from run_monitor_models import metrics,bootstrap

PRIMARY='regional_kernel'


def stages(p):
    return {'discovery_2019':p.development_city&p.year.eq(2019),
            'validation_2020':p.development_city&p.year.eq(2020),
            'temporal_audit':p.development_city&p.year.ge(2021),
            'new_city_audit':~p.development_city&p.year.ge(2021),
            'new_grid_audit':~p.development_city&p.new_weather_grid&p.year.ge(2021)}


def folder_for(cohort,direction,seed,width,method):
    root=WINDOWS/f'window_{width}' if method=='fixed' else HERE/f'valid_{width}'
    return root/f'results/{cohort}/{direction}/{seed}'


def main():
    assert json.loads((HERE/'audit/model_completion.json').read_text())['new_gap_configurations']==24
    metric_rows=[];primary=[];budgets=[];availability=[];paired_models=[];paired_primary=[]
    for cohort in old.COHORTS:
        for direction in ['CHN_to_AUS','AUS_to_CHN']:
            for seed in old.SEEDS:
                full=read(WINDOWS/f'window_12/results/{cohort}/{direction}/{seed}/predictions.csv.gz')
                for width in [3,1]:
                    predictions={k:read(folder_for(cohort,direction,seed,width,k)/'predictions.csv.gz') for k in ['fixed','valid']}
                    keys={k:set(zip(p.eFUA_ID,p.t)) for k,p in predictions.items()}
                    common=keys['fixed']&keys['valid'];added=keys['valid']-keys['fixed']
                    for method,p in predictions.items():
                        folder=folder_for(cohort,direction,seed,width,method)
                        audit=read(folder/'fit_audit.csv');labels=read(folder/'calibration_rows.csv.gz')
                        setting={'cohort':cohort,'direction':direction,'seed':seed,'width':width,'method':method}
                        budget={**setting,'fits':len(audit),'skipped_prediction_months':48-len(audit),
                            'labels_per_fit_min':audit.labels.min(),'labels_per_fit_median':audit.labels.median(),
                            'labels_per_fit_max':audit.labels.max(),'unique_labels_across_run':len(labels),
                            'training_cities':labels.eFUA_ID.nunique(),'predicted_rows':len(p),
                            'full_eligible_rows':len(full),'added_rows':len(added) if method=='valid' else 0,
                            'lost_rows':len(keys['fixed']-keys['valid']) if method=='valid' else 0}
                        if method=='valid':
                            h=read(folder/'history_audit.csv');used=h[h.fit_performed]
                            budget.update({'max_label_age':used.oldest_age.max(),
                                'fits_using_older_labels':int(used.labels_older_than_nominal_window.gt(0).sum()),
                                'label_reuses_outside_nominal_window':int(used.labels_older_than_nominal_window.sum())})
                        budgets.append(budget)
                        masks={'native':np.ones(len(p),bool),
                            'common':np.array([(c,t) in common for c,t in zip(p.eFUA_ID,p.t)])}
                        if method=='valid':masks['added']=np.array([(c,t) in added for c,t in zip(p.eFUA_ID,p.t)])
                        for stage,use in stages(p).items():
                            denom=full[stages(full)[stage]]
                            for selection,mask in masks.items():
                                frame=p[use&mask];fraction=len(frame)/len(denom) if len(denom) else 0
                                availability.append({**setting,'stage':stage,'selection':selection,'rows':len(frame),
                                    'eligible_stage_rows':len(denom),'coverage':fraction,
                                    'fuas':frame.eFUA_ID.nunique(),'months':frame.t.nunique()})
                                mm={m:metrics(frame,frame[m]) for m in old.MODELS}
                                for m,result in mm.items():metric_rows.append({**setting,'stage':stage,'selection':selection,'model':m,**result})
                                local=min(old.LOCAL,key=lambda m:mm[m]['rmse'] if mm[m]['rmse'] is not None else np.inf)
                                comp=min(old.REFS,key=lambda m:mm[m]['rmse'] if mm[m]['rmse'] is not None else np.inf)
                                rms=mm[PRIMARY]['rmse'];ref=mm[comp]['rmse'];lr=mm[local]['rmse']
                                gain=1-rms/max(ref,1e-12) if rms is not None else np.nan
                                lg=1-rms/max(lr,1e-12) if rms is not None else np.nan
                                expected=set(denom.year.unique());support=fraction>=.5 and set(frame.year.unique())==expected
                                for year in expected:
                                    z=frame[frame.year.eq(year)]
                                    support &= len(z)>=24 and z.eFUA_ID.nunique()>=3 and z.t.nunique()>=6
                                cis={}
                                if 'audit' in stage and selection!='added':
                                    support &= len(expected)>=2
                                    for cluster in ['eFUA_ID','weather_grid_id']:
                                        for reference in old.REFS:
                                            lo,hi=bootstrap(frame,PRIMARY,reference,seed,cluster)
                                            cis[f'{cluster}_{reference}_low']=lo;cis[f'{cluster}_{reference}_high']=hi
                                lows=[v for k,v in cis.items() if k.endswith('_low')]
                                boot=bool(lows) and all(v is not None and np.isfinite(v) and v>0 for v in lows)
                                primary.append({**setting,'stage':stage,'selection':selection,'model':PRIMARY,
                                    'gain':gain,'gain_vs_local':lg,'best_comparator':comp,'best_local':local,
                                    'comparator_rmse':ref,'local_rmse':lr,'coverage':fraction,'support_pass':bool(support),
                                    'bootstrap_pass':boot,'screen_pass':bool(selection!='added' and support and boot and gain>=.05),
                                    **mm[PRIMARY],**cis})
                    # Paired deltas compare the same target labels, including the same row order.
                    a=predictions['fixed'].set_index(['eFUA_ID','t']).loc[sorted(common)].reset_index()
                    b=predictions['valid'].set_index(['eFUA_ID','t']).loc[sorted(common)].reset_index()
                    assert np.array_equal(a.observed_pm25,b.observed_pm25)
                    for stage,use in stages(a).items():
                        x=a[use].copy();y=b[use]
                        for model in old.MODELS:
                            before=metrics(x,x[model]);after=metrics(y,y[model])
                            paired_models.append({'cohort':cohort,'direction':direction,'seed':seed,'width':width,'stage':stage,
                                'model':model,'n':len(x),'rmse_fixed':before['rmse'],'rmse_valid':after['rmse'],
                                'delta_rmse_valid_minus_fixed':after['rmse']-before['rmse'] if len(x) else np.nan})
                        row=paired_models[-len(old.MODELS)+old.MODELS.index(PRIMARY)].copy()
                        x['fixed_primary']=x[PRIMARY].to_numpy();x['valid_primary']=y[PRIMARY].to_numpy()
                        for cluster in ['eFUA_ID','weather_grid_id']:
                            lo,hi=bootstrap(x,'valid_primary','fixed_primary',seed,cluster)
                            row[f'{cluster}_rmse_improvement_low']=lo;row[f'{cluster}_rmse_improvement_high']=hi
                        paired_primary.append(row)
                print('EVALUATED',cohort,direction,seed,flush=True)
    for name,rows in [('metrics.csv.gz',metric_rows),('primary.csv',primary),('label_accounting.csv',budgets),
        ('availability.csv',availability),('paired_models.csv.gz',paired_models),('paired_primary.csv',paired_primary)]:
        pd.DataFrame(rows).to_csv(HERE/'audit'/name,index=False)
    pr=pd.DataFrame(primary);joint=[]
    for key,g in pr[pr.selection.eq('common')].groupby(['cohort','direction','seed','width','method']):
        audit=g[g.stage.isin(['temporal_audit','new_city_audit'])];assert len(audit)==2
        joint.append(dict(zip(['cohort','direction','seed','width','method'],key))|{'joint_pass':bool(audit.screen_pass.all())})
    pd.DataFrame(joint).to_csv(HERE/'audit/joint_screens.csv',index=False)
    dump(HERE/'audit/evaluation_complete.json',{'completed_at':now(),'metric_rows':len(metric_rows),
        'primary_rows':len(primary),'paired_metric_rows':len(paired_models),'primary_width':3,'primary_model':PRIMARY,
        'fresh_external_confirmation':False,'targets_imputed':False})


if __name__=='__main__':main()
