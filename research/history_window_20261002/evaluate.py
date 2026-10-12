"""Same-row window comparisons, accounting and primary-model screens."""
import json
import numpy as np
import pandas as pd
from run_windows import HERE,CORRECTED,PREVIOUS,WINDOWS,PRIMARY,old,read,dump,sha,now
from run_monitor_models import metrics,bootstrap

def stages(p):
    return {'discovery_2019':p.development_city&p.year.eq(2019),
            'validation_2020':p.development_city&p.year.eq(2020),
            'temporal_audit':p.development_city&p.year.ge(2021),
            'new_city_audit':~p.development_city&p.year.ge(2021),
            'new_grid_audit':~p.development_city&p.new_weather_grid&p.year.ge(2021)}

def main():
    assert json.loads((HERE/'audit/completion.json').read_text())['configurations']==48
    metric_rows=[];primary=[];budgets=[];availability=[]
    for cohort in old.COHORTS:
        panel=read(CORRECTED/f'data/{cohort}_panel.csv.gz')
        for direction in ['CHN_to_AUS','AUS_to_CHN']:
            for seed in old.SEEDS:
                predictions={};audits={}
                for w in WINDOWS:
                    folder=HERE/f'window_{w}/results/{cohort}/{direction}/{seed}'
                    p=read(folder/'predictions.csv.gz');predictions[w]=p
                    audits[w]=read(folder/'fit_audit.csv')
                    labels=read(folder/'calibration_rows.csv.gz')
                    budgets.append({'cohort':cohort,'direction':direction,'seed':seed,'window':w,
                        'fits':len(audits[w]),'skipped_prediction_months':48-len(audits[w]),
                        'labels_per_fit_min':audits[w].labels.min(),'labels_per_fit_median':audits[w].labels.median(),
                        'labels_per_fit_max':audits[w].labels.max(),'unique_labels_across_run':len(labels),
                        'training_cities':labels.eFUA_ID.nunique(),'predicted_rows':len(p)})
                full=predictions[12]
                keys=set.intersection(*(set(zip(p.eFUA_ID,p.t)) for p in predictions.values()))
                for w,p in predictions.items():
                    common=np.array([(city,t) in keys for city,t in zip(p.eFUA_ID,p.t)])
                    setting={'cohort':cohort,'direction':direction,'seed':seed,'window':w}
                    for stage,base_mask in stages(full).items():
                        denominator=full[base_mask]
                        native=p[stages(p)[stage]]
                        for selection,frame in [('native',native),('common',p[stages(p)[stage]&common])]:
                            fraction=len(frame)/len(denominator) if len(denominator) else 0
                            availability.append({**setting,'stage':stage,'selection':selection,'rows':len(frame),
                                'full_12month_stage_rows':len(denominator),'coverage':fraction,
                                'fuas':frame.eFUA_ID.nunique(),'months':frame.t.nunique()})
                            mm={model:metrics(frame,frame[model]) for model in old.MODELS}
                            for model,m in mm.items():metric_rows.append({**setting,'stage':stage,'selection':selection,'model':model,**m})
                            local=min(old.LOCAL,key=lambda m:mm[m]['rmse'] if mm[m]['rmse'] is not None else np.inf)
                            comparator=min(old.REFS,key=lambda m:mm[m]['rmse'] if mm[m]['rmse'] is not None else np.inf)
                            rms=mm[PRIMARY]['rmse'];ref=mm[comparator]['rmse'];local_rms=mm[local]['rmse']
                            gain=1-rms/max(ref,1e-12) if rms is not None else np.nan
                            local_gain=1-rms/max(local_rms,1e-12) if rms is not None else np.nan
                            expected=set(denominator.year.unique())
                            support=fraction>=.5 and expected==set(frame.year.unique())
                            for year in expected:
                                z=frame[frame.year.eq(year)]
                                support &= len(z)>=24 and z.eFUA_ID.nunique()>=3 and z.t.nunique()>=6
                            cis={}
                            if 'audit' in stage:
                                support &= len(expected)>=2
                                for cluster in ['eFUA_ID','weather_grid_id']:
                                    for comp in old.REFS:
                                        lo,hi=bootstrap(frame,PRIMARY,comp,seed,cluster)
                                        cis[f'{cluster}_{comp}_low']=lo;cis[f'{cluster}_{comp}_high']=hi
                            lows=[v for k,v in cis.items() if k.endswith('_low')]
                            boot=bool(lows) and all(v is not None and np.isfinite(v) and v>0 for v in lows)
                            primary.append({**setting,'selection':selection,'stage':stage,'model':PRIMARY,
                                'gain':gain,'gain_vs_local':local_gain,'best_comparator':comparator,'best_local':local,
                                'comparator_rmse':ref,'local_rmse':local_rms,'coverage':fraction,'support_pass':bool(support),
                                'bootstrap_pass':boot,'screen_pass':bool(support and boot and gain>=.05),
                                **mm[PRIMARY],**cis})
                print('EVALUATED',cohort,direction,seed,flush=True)
    for name,rows in [('metrics.csv.gz',metric_rows),('primary.csv',primary),('label_accounting.csv',budgets),('availability.csv',availability)]:
        pd.DataFrame(rows).to_csv(HERE/'audit'/name,index=False)
    p=pd.DataFrame(primary);joint=[]
    for key,g in p[p.selection.eq('common')].groupby(['cohort','direction','seed','window']):
        audit=g[g.stage.isin(['temporal_audit','new_city_audit'])]
        assert len(audit)==2
        joint.append(dict(zip(['cohort','direction','seed','window'],key))|{'joint_pass':bool(audit.screen_pass.all())})
    pd.DataFrame(joint).to_csv(HERE/'audit/joint_screens.csv',index=False)
    common=p[p.selection.eq('common')]
    keys=['cohort','direction','seed','stage']
    paired=common[common.window.eq(3)].merge(common[common.window.eq(12)],on=keys,validate='one_to_one',suffixes=('_3','_12'))
    assert paired.n_3.eq(paired.n_12).all()
    paired['delta_rmse']=paired.rmse_3-paired.rmse_12
    paired['delta_gain']=paired.gain_3-paired.gain_12
    paired.to_csv(HERE/'audit/primary_3_vs_12.csv',index=False)
    dump(HERE/'audit/evaluation_complete.json',{'created_at':now(),'metric_rows':len(metric_rows),
        'primary_rows':len(primary),'joint_rows':len(joint),'primary_window':3,'primary_model':PRIMARY,
        'fresh_confirmation':False,'fixed_global_label_budget':False})

if __name__=='__main__':main()
