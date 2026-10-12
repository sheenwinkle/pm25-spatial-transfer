"""Equal-budget observed-label transfer with discovery-frozen conditional rules."""
import os
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import argparse
import itertools
import json
import time
import warnings
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, KernelCenterer
from common import HERE, DATA, AUDIT, RESULTS, dump, digest

Y='observed_pm25'
SOURCE_END=2018
DISC_FIRST=2019
AUDIT_FIRST=2021
AUDIT_END=2022
LOCAL=['target_mean','target_ridge','target_rbf','target_rf','target_extra','target_lgbm']
SOURCE=['source_affine','source_residual','source_kernel','weighted_source_kernel']
NEG=[f'shuffled_kernel_{s}' for s in (5993,5994,5995)]
MODELS=LOCAL+SOURCE+NEG+['source_only']
GATES=['temperature_c','humidity_pct','wind_mps','monthly_power_precip_mm_day',
       'distance_to_coast_km','log_fire_count_100km','log_population_density','support_distance']

def pos(v):return np.maximum(np.asarray(v,float),0)

def norm():return make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler())

def tree(source=False):
    return lgb.LGBMRegressor(n_estimators=128 if source else 100,learning_rate=.04 if source else .05,
        num_leaves=15 if source else 7,min_child_samples=40 if source else 5,
        reg_lambda=10 if source else 5,n_jobs=4,random_state=5993,verbosity=-1)

def embed(model,x):
    a=model.predict(x,pred_leaf=True).astype(np.int32)
    if a.ndim==1:a=a[:,None]
    n,k=a.shape;assert a.min()>=0 and a.max()<15
    return sparse.csr_matrix((np.full(n*k,1/np.sqrt(k)),(a+15*np.arange(k)).ravel(),np.arange(n+1)*k),shape=(n,k*15))

def kernel(kk,kt,y):
    c=KernelCenterer();km=c.fit_transform(kk)
    f=KernelRidge(alpha=1,kernel='precomputed').fit(km,y-y.mean())
    return pos(y.mean()+f.predict(c.transform(kt)))

def metrics(d,p):
    if len(d)==0:return {'n':0,'fuas':0,'months':0,'rmse':None,'mae':None,'r2':None,'bias':None}
    y=d[Y].to_numpy();p=np.asarray(p);e=p-y;v=np.sum((y-y.mean())**2)
    return {'n':len(d),'fuas':d.eFUA_ID.nunique(),'months':d.t.nunique(),
        'rmse':float(np.sqrt(np.mean(e*e))),'mae':float(np.mean(abs(e))),
        'r2':float(1-np.sum(e*e)/v) if v>0 else None,'bias':float(e.mean())}

def maskrule(d,rule):
    m=np.ones(len(d),bool)
    for a in rule:
        v=d[a['feature']].to_numpy()
        m &= v<=a['threshold'] if a['op']=='<=' else v>=a['threshold']
    return m

def gain(d,p,mask):
    e=(d.loc[mask,MODELS].to_numpy()-d.loc[mask,Y].to_numpy()[:,None])**2
    rms=dict(zip(MODELS,np.sqrt(e.mean(0))))
    best=min(LOCAL+NEG,key=rms.get)
    return 1-rms[p]/max(rms[best],1e-12),best,rms

def supported(d,m):
    if not len(d) or not m.any():return False
    return (m.mean()>=.25 and d.loc[m,'eFUA_ID'].nunique()>=max(3,np.ceil(.25*d.eFUA_ID.nunique()))
            and d.loc[m,'t'].nunique()>=12)

def discover(d):
    atoms=[]
    for f in GATES:
        v=d[f].to_numpy(float)
        if np.isfinite(v).sum()<20:continue
        for q in np.unique(np.nanquantile(v,[.1,.25,.5,.75,.9])):
            for op in ['<=','>=']:
                a={'feature':f,'op':op,'threshold':float(q)};m=maskrule(d,[a])
                if supported(d,m) and m.mean()<.99:atoms.append(a)
    rows=[]
    for model in SOURCE:
        singles=[]
        for rule in [[]]+[[a] for a in atoms]:
            m=maskrule(d,rule)
            if not supported(d,m):continue
            g,ref,_=gain(d,model,m)
            singles.append((g,rule,ref))
        top=sorted([a for a in singles if a[1]],key=lambda a:a[0],reverse=True)[:12]
        candidates=[a[1] for a in singles]+[a[1]+b[1] for a,b in itertools.combinations(top,2)]
        for rule in candidates:
            m=maskrule(d,rule)
            if not supported(d,m):continue
            g,ref,_=gain(d,model,m)
            annual=[]
            for year in sorted(d.year.unique()):
                z=d[d.year.eq(year)];mz=maskrule(z,rule)
                if not mz.any():annual.append(False);continue
                err=(z.loc[mz,model]-z.loc[mz,Y])**2;re=(z.loc[mz,ref]-z.loc[mz,Y])**2
                annual.append(bool(err.mean()<re.mean()))
            rows.append({'model':model,'rule':rule,'reference':ref,'gain':g,'coverage':float(m.mean()),
                'complexity':len(rule),'discovery_pass':bool(g>=.05 and all(annual))})
    if not rows:raise ValueError('Not enough development cities/months for conditional discovery')
    passed=[r for r in rows if r['discovery_pass']]
    chosen=sorted(passed,key=lambda r:(r['complexity'],-r['coverage'],-r['gain']))[0] if passed else max(rows,key=lambda r:r['gain'])
    return chosen,rows

def bootstrap(d,model,reference,seed,cluster):
    if d.eFUA_ID.nunique()<2 or d.quarter_id.nunique()<2:return [None,None]
    ci,_=pd.factorize(d[cluster]);ti,_=pd.factorize(d.quarter_id);nc,nt=ci.max()+1,ti.max()+1
    if nc<2:return [None,None]
    count=np.zeros((nc,nt));s1=count.copy();s2=count.copy()
    np.add.at(count,(ci,ti),1);np.add.at(s1,(ci,ti),(d[model]-d[Y])**2);np.add.at(s2,(ci,ti),(d[reference]-d[Y])**2)
    rng=np.random.default_rng(seed);cw=rng.multinomial(nc,np.full(nc,1/nc),1000);tw=rng.multinomial(nt,np.full(nt,1/nt),1000)
    den=np.sum((cw@count)*tw,axis=1);ok=den>0
    delta=np.sqrt((np.sum((cw@s2)*tw,axis=1)/np.maximum(den,1))[ok])-np.sqrt((np.sum((cw@s1)*tw,axis=1)/np.maximum(den,1))[ok])
    return np.quantile(delta,[.025,.975]).tolist()

def run(d,features,cohort,feature_set,direction):
    src,tgt=direction.split('_to_');s=d[d.country_iso.eq(src)&d.year.le(SOURCE_END)].copy()
    t=d[d.country_iso.eq(tgt)&d.year.le(AUDIT_END)].copy().reset_index(drop=True)
    folder=RESULTS/cohort/feature_set/direction;folder.mkdir(parents=True,exist_ok=True)
    base=tree(True).fit(s[features],s[Y]);models={'source':base};embeddings={'source_kernel':embed(base,t[features])}
    for seed in (5993,5994,5995):
        neg=tree(True).fit(s[features],np.random.default_rng(seed).permutation(s[Y].to_numpy()))
        models[f'shuffle_{seed}']=neg;embeddings[f'shuffled_kernel_{seed}']=embed(neg,t[features])
    t['source_only']=pos(base.predict(t[features]));normal=norm();sx=normal.fit_transform(s[features[:6]])
    t['support_distance']=cKDTree(sx).query(normal.transform(t[features[:6]]),workers=4)[0]
    eligible=t[t.year.eq(SOURCE_END)].groupby('eFUA_ID').size();eligible=sorted(eligible[eligible>=6].index)
    assert len(eligible)>=5
    metrics_rows=[];gates=[];budgets=[];splits=[]
    for seed in (5993,5994,5995):
        rng=np.random.default_rng(seed);ids=np.array(eligible).copy();rng.shuffle(ids);ids=ids[:min(10,int(np.ceil(len(ids)/2)))]
        dev=t.eFUA_ID.isin(ids).to_numpy();devgrids=set(t.loc[dev,'weather_grid_id'])
        for city in sorted(t.eFUA_ID.unique()):splits.append({'seed':seed,'eFUA_ID':city,'development':city in ids,'eligible_pre2019':city in eligible})
        u=t[dev&t.year.le(SOURCE_END).to_numpy()]
        dn=norm();dx=dn.fit_transform(pd.concat([s[features],u[features]],ignore_index=True))
        clf=LogisticRegression(C=.1,class_weight='balanced',max_iter=1000).fit(dx,np.r_[np.zeros(len(s)),np.ones(len(u))])
        prob=clf.predict_proba(dx[:len(s)])[:,1];weights=np.clip(prob/np.maximum(1-prob,1e-9),.1,10)
        weighted=tree(True).fit(s[features],s[Y],sample_weight=weights)
        embeddings['weighted_source_kernel']=embed(weighted,t[features]);models[f'weighted_{seed}']=weighted
        dump(folder/f'weighting_{seed}.json',{'min':weights.min(),'max':weights.max(),'effective_source_rows':weights.sum()**2/(weights**2).sum(),
            'source_rows':len(s),'unlabelled_target_rows':len(u),'development_fuas':list(ids)})
        for w in (3,6,12):
            p=t.copy()
            for m in MODELS:
                if m!='source_only':p[m]=np.nan
            p['fit_labels']=0;used=np.zeros(len(t),bool);fit_audit=[]
            for month in sorted(t.loc[t.year.ge(DISC_FIRST),'t'].unique()):
                fit=dev&t.t.between(month-w,month-1).to_numpy();ev=t.t.eq(month).to_numpy()
                if fit.sum()<5:continue
                assert not (fit&ev).any() and t.loc[fit,'t'].max()<month and not (fit&~dev).any()
                used|=fit;y=t.loc[fit,Y].to_numpy();x=t.loc[fit,features];ex=t.loc[ev,features]
                p.loc[ev,'fit_labels']=fit.sum()
                fit_audit.append({'month_t':int(month),'labels':int(fit.sum()),'latest_fit_t':int(t.loc[fit,'t'].max()),'fit_fuas':int(t.loc[fit,'eFUA_ID'].nunique())})
                p.loc[ev,'target_mean']=y.mean()
                z=norm();xx=z.fit_transform(x);ee=z.transform(ex)
                p.loc[ev,'target_ridge']=pos(Ridge(alpha=10).fit(xx,y).predict(ee))
                p.loc[ev,'target_rbf']=kernel(rbf_kernel(xx),rbf_kernel(ee,xx),y)
                for key,cls in [('target_rf',RandomForestRegressor),('target_extra',ExtraTreesRegressor)]:
                    m=cls(n_estimators=80,min_samples_leaf=3,max_features=1.0,n_jobs=4,random_state=5993).fit(xx,y)
                    p.loc[ev,key]=pos(m.predict(ee))
                p.loc[ev,'target_lgbm']=pos(tree().fit(x,y).predict(ex))
                raw=t.loc[fit,'source_only'].to_numpy();eraw=t.loc[ev,'source_only'].to_numpy()
                zz=StandardScaler();rx=zz.fit_transform(raw[:,None]);re=zz.transform(eraw[:,None])
                p.loc[ev,'source_affine']=pos(Ridge(alpha=10).fit(rx,y).predict(re))
                p.loc[ev,'source_residual']=pos(eraw+tree().fit(x,y-raw).predict(ex))
                for name,e in embeddings.items():
                    a=e[fit];b=e[ev];p.loc[ev,name]=kernel((a@a.T).toarray(),(b@a.T).toarray(),y)
            p['development_city']=dev;p['new_weather_grid']=~t.weather_grid_id.isin(devgrids)
            p=p[p.year.ge(DISC_FIRST)].copy();good=p[MODELS].notna().all(axis=1)
            missing=p[~good];missing[['eFUA_ID','year','month','fit_labels']].to_csv(folder/f'no_prediction_{seed}_{w}.csv',index=False)
            p=p[good].reset_index(drop=True)
            disc=p[p.development_city&p.year.between(DISC_FIRST,AUDIT_FIRST-1)]
            lock,search=discover(disc);p['accepted']=maskrule(p,lock['rule']);p['selected_prediction']=p[lock['model']]
            dump(folder/f'lock_{seed}_{w}.json',lock)
            pd.DataFrame(search).assign(rule=lambda f:f.rule.map(json.dumps)).to_csv(folder/f'search_{seed}_{w}.csv.gz',index=False)
            setting={'cohort':cohort,'feature_set':feature_set,'direction':direction,'seed':seed,'window_months':w}
            budgets.append({**setting,'development_fuas':len(ids),'min_labels_per_fit':p.fit_labels.min(),'max_labels_per_fit':p.fit_labels.max(),
                'cumulative_unique_labels':int(used.sum()),'discovery_label_rows':len(disc)})
            pd.DataFrame(fit_audit).to_csv(folder/f'fit_audit_{seed}_{w}.csv',index=False)
            stages={'discovery':p.development_city&p.year.lt(AUDIT_FIRST),'temporal_audit':p.development_city&p.year.ge(AUDIT_FIRST),
                'new_city_audit':~p.development_city&p.year.ge(AUDIT_FIRST),
                'new_weather_grid_audit':~p.development_city&p.new_weather_grid&p.year.ge(AUDIT_FIRST)}
            for stage,m in stages.items():
                full=p[m];sel=full[full.accepted]
                for selection,rows in [('full',full),('accepted',sel)]:
                    for model in MODELS:
                        metrics_rows.append({**setting,'stage':stage,'selection':selection,'model':model,**metrics(rows,rows[model])})
                if len(sel):
                    g,best,_=gain(full,lock['model'],full.accepted)
                    refs=list(dict.fromkeys([lock['reference'],'target_mean']+NEG));cis={}
                    for cluster in ['eFUA_ID','weather_grid_id']:
                        for ref in refs:
                            lo,hi=bootstrap(sel,lock['model'],ref,seed,cluster);cis[f'{cluster}_{ref}_low']=lo;cis[f'{cluster}_{ref}_high']=hi
                    okci=all(v is not None and v>0 for k,v in cis.items() if k.endswith('_low'))
                    gates.append({**setting,'stage':stage,'selected_model':lock['model'],'rule':json.dumps(lock['rule']),
                        'discovery_pass':lock['discovery_pass'],'coverage':len(sel)/len(full),'relative_gain':g,'best_comparator':best,
                        'support_pass':supported(full,full.accepted.to_numpy()),'bootstrap_pass':okci,
                        'pass':bool(lock['discovery_pass'] and g>=.05 and supported(full,full.accepted.to_numpy()) and okci),
                        **metrics(sel,sel[lock['model']]),**cis})
            keep=['eFUA_ID','fua_name','year','month','t','quarter_id','weather_grid_id',Y,'station_count','sites',
                'development_city','new_weather_grid','fit_labels','accepted','selected_prediction']+GATES+MODELS
            p[list(dict.fromkeys(keep))].to_csv(folder/f'predictions_{seed}_{w}.csv.gz',index=False)
            print('MONITOR_CONFIG',cohort,feature_set,direction,seed,w,'discovery_gain',round(lock['gain'],4),flush=True)
    joblib.dump({'models':models,'features':features},folder/'source_models.joblib',compress=3)
    pd.DataFrame(metrics_rows).to_csv(folder/'metrics.csv',index=False)
    pd.DataFrame(gates).to_csv(folder/'gates.csv',index=False)
    pd.DataFrame(budgets).to_csv(folder/'budgets.csv',index=False)
    pd.DataFrame(splits).to_csv(folder/'city_splits.csv',index=False)
    dump(folder/'manifest.json',{'source_rows':len(s),'source_fuas':s.eFUA_ID.nunique(),'target_rows':len(t),
        'features':features,'source_end':SOURCE_END,'discovery_first':DISC_FIRST,'audit_first':AUDIT_FIRST,
        'audit_end':AUDIT_END,'protocol_sha256':digest(HERE/'EXTENSION_PROTOCOL.md'),'script_sha256':digest(__file__),'status':'complete'})

def main():
    global SOURCE_END,DISC_FIRST,AUDIT_FIRST,AUDIT_END
    ap=argparse.ArgumentParser();ap.add_argument('--experiment',choices=['outer_time','mixing'],required=True);args=ap.parse_args()
    features=json.loads((AUDIT/'build_summary.json').read_text())['features']
    d=pd.read_csv(DATA/'all_networks_model_panel.csv.gz');cohort='all_networks'
    if args.experiment=='outer_time':
        SOURCE_END=2016;DISC_FIRST=2017;AUDIT_FIRST=2019;AUDIT_END=2020
        fs='outer_time_2017_2018'
    else:
        mix=pd.read_csv(DATA/'common_era5_month_2015_2022.csv.gz')
        cols=['era5_pbl_mean_m','era5_pbl_low_fraction','era5_surface_ventilation_mean_m2_s']
        d=d.merge(mix[['eFUA_ID','year','month']+cols],on=['eFUA_ID','year','month'],validate='one_to_one')
        assert d[cols].notna().all().all()
        for c in [cols[0],cols[2]]:d['log_'+c]=np.log1p(d[c]);features.append('log_'+c)
        features.append(cols[1]);fs='monthly_plus_mixing'
    for direction in ['CHN_to_AUS','AUS_to_CHN']:
        if (RESULTS/cohort/fs/direction/'manifest.json').exists():continue
        run(d,features,cohort,fs,direction)

if __name__=='__main__':main()
