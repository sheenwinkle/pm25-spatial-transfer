from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

import run_study as b

HERE=Path(__file__).resolve().parent
OUT=HERE/"anomaly_results"
P=json.loads((HERE/"anomaly_protocol.json").read_text(encoding="utf-8"))
FEATURES=["surface_temperature_mean_c","surface_relative_humidity_mean_pct",
          "surface_wind_mean_mps","surface_dry_visibility_harmonic_mean_km",
          "surface_visibility_harmonic_mean_km","surface_stagnant_day_fraction",
          "surface_precipitation_sum_mm","surface_sea_level_pressure_mean_hpa",
          "local_season_sin","local_season_cos"]


def weather_valid(t):
    return (t.temperature_c.between(-80,60)&t.humidity_pct.between(0,100)&t.wind_mps.between(0,80)).to_numpy()


def source_fit(d,source,target):
    s=d[(d.country_iso==source)&(d.year<=2018)].copy()
    s=s.loc[weather_valid(s)].reset_index(drop=True)
    assert s.eFUA_ID.nunique()>=5
    t=d[(d.country_iso==target)&(d.year>=2017)].copy().reset_index(drop=True)
    s["log_y"]=np.log1p(s[b.Y])
    city=s.groupby("eFUA_ID").log_y
    citymonth=s.groupby(["eFUA_ID","month"]).log_y
    period=s.groupby(["year","month"]).log_y
    cityres=s.log_y-city.transform("mean")
    seasonres=s.log_y-citymonth.transform("mean")
    season_sd=seasonres.groupby(s.eFUA_ID).transform("std").clip(lower=.05).fillna(.05)
    labels={"source_log_level":s.log_y,
            "source_city_z":cityres/city.transform("std").clip(lower=.05).fillna(.05),
            "source_city_month_z":seasonres/season_sd,
            "source_period_z":(s.log_y-period.transform("mean"))/period.transform("std").clip(lower=.05).fillna(.05)}
    preds={};models={}
    for name,y in labels.items():
        m=lgb.LGBMRegressor(**P["source_lightgbm"],random_state=5993,n_jobs=4,verbosity=-1)
        m.fit(s[FEATURES],y)
        preds[name]=m.predict(t[FEATURES])
        models[name]=m
    norm=make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler())
    sx=norm.fit_transform(s[FEATURES]);tx=norm.transform(t[FEATURES])
    rng=np.random.default_rng(5993)
    ids=np.array(sorted(s.eFUA_ID.unique()));rng.shuffle(ids)
    ref=s.eFUA_ID.isin(ids[:int(.8*len(ids))]).to_numpy()
    ridx=np.flatnonzero(ref);cidx=np.flatnonzero(~ref)
    ridx=rng.choice(ridx,min(len(ridx),5000),replace=False)
    cidx=rng.choice(cidx,min(len(cidx),1500),replace=False)
    tree=cKDTree(sx[ridx])
    t["support_distance"]=tree.query(tx,k=1,workers=4)[0]
    threshold=float(np.quantile(tree.query(sx[cidx],k=1,workers=4)[0],.95))
    diagnostics={"direction":source+"_to_"+target,"source_rows":len(s),"source_fuas":s.eFUA_ID.nunique(),
                 "features":FEATURES,"source_cross_fua_distance_95":threshold,
                 "target_complete_weather_fraction_2023_2024":float(weather_valid(t[t.year>=2023]).mean())}
    b.dump(OUT/f"source_{source}_diagnostics.json",diagnostics)
    joblib.dump({"models":models,"features":FEATURES,"source_scaler":norm},OUT/f"source_{source}_models.joblib",compress=3)
    print("anomaly_source_done",diagnostics,flush=True)
    return t,preds


def search(frame,pred,local):
    eligible=weather_valid(frame)
    atoms=b.masks_for_atoms(frame)
    y=frame[b.Y].to_numpy()
    records=[]
    def one(model,rule,mask):
        mask=mask & eligible
        cities=frame.loc[mask,"eFUA_ID"].nunique()
        if mask.mean()<.25 or cities<5 or frame.loc[mask,"t"].nunique()<12:return None
        err=float(np.sqrt(np.mean((pred[model][mask]-y[mask])**2)))
        errs={k:float(np.sqrt(np.mean((v[mask]-y[mask])**2))) for k,v in local.items()}
        reference=min(errs,key=errs.get)
        gain=1-err/max(errs[reference],1e-12)
        annual=[]
        for year in sorted(frame.year.unique()):
            m=mask&(frame.year.to_numpy()==year)
            annual.append(float(np.sqrt(np.mean((local[reference][m]-y[m])**2))-np.sqrt(np.mean((pred[model][m]-y[m])**2))) if m.any() else -np.inf)
        r={"model":model,"baseline":reference,"rule":rule,"learned_conditions":len(rule),
           "mandatory_weather_eligibility":True,"coverage":float(mask.mean()),"rmse":err,
           "baseline_rmse":errs[reference],"relative_improvement":gain,
           "discovery_pass":bool(gain>=.05 and min(annual)>0)}
        records.append(r)
        return r
    for model in pred:
        one(model,[],np.ones(len(frame),bool))
        singles=[]
        for i,(atom,mask) in enumerate(atoms):
            r=one(model,[atom],mask)
            if r:singles.append((r["relative_improvement"],i))
        for i,j in itertools.combinations([i for _,i in sorted(singles,reverse=True)[:12]],2):
            a,ma=atoms[i];c,mc=atoms[j]
            one(model,[a,c],ma&mc)
    valid=[r for r in records if r["discovery_pass"]]
    if valid:
        chosen=sorted(valid,key=lambda r:(r["learned_conditions"],-r["coverage"],-r["relative_improvement"]))[0]
    elif records:
        chosen=max((r for r in records if r["learned_conditions"]==0),key=lambda r:r["relative_improvement"])
    else:
        mask=eligible
        errors={k:np.sqrt(np.mean((v[mask]-y[mask])**2)) for k,v in pred.items()}
        refs={k:np.sqrt(np.mean((v[mask]-y[mask])**2)) for k,v in local.items()}
        chosen={"model":min(errors,key=errors.get),"baseline":min(refs,key=refs.get),"rule":[],
                "learned_conditions":0,"mandatory_weather_eligibility":True,"coverage":float(mask.mean()),
                "discovery_pass":False,"reason":"insufficient weather-observed development support"}
    return chosen,records


def inverse_prediction(model,x,ex,logy):
    train_pred=model.predict(x)
    smear=float(np.exp(logy-train_pred).mean())
    return np.maximum(np.exp(model.predict(ex))*smear-1,0)


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    d,_,provenance=joblib.load(HERE/"fua_month_cache.joblib")
    metrics=[];allmodels=[];locks=[];grid=[];coef=[]
    for direction in ["CHN_to_AUS","AUS_to_CHN"]:
        source,target=direction.split("_to_")
        t,sp=source_fit(d,source,target)
        manifest=pd.read_csv(b.OUT/f"city_manifest_{direction}.csv")
        for seed in P["city_partitions"]:
            dev_ids=manifest.loc[(manifest.seed==seed)&manifest.development_city,"eFUA_ID"].to_numpy(copy=True)
            rng=np.random.default_rng(seed);rng.shuffle(dev_ids)
            dev=t.eFUA_ID.isin(dev_ids).to_numpy()
            stages={"discovery":dev&t.year.between(2019,2020).to_numpy(),
                    "confirmation":dev&t.year.between(2021,2022).to_numpy(),
                    "spatial_temporal_audit":~dev&t.year.between(2023,2024).to_numpy(),
                    "seen_city_temporal_audit":dev&t.year.between(2023,2024).to_numpy()}
            df=t.loc[stages["discovery"]].copy()
            for count in P["sentinel_city_counts"]:
                sentinel=t.eFUA_ID.isin(dev_ids[:count]).to_numpy()
                for window in P["target_windows_months"]:
                    names=list(sp)+["target_mean","target_log_ridge","target_log_lgbm"]
                    preds={name:np.full(len(t),np.nan) for name in names}
                    used=np.zeros(len(t),bool)
                    for month in sorted(t.loc[t.year>=2019,"t"].unique()):
                        fit=sentinel&t.t.between(month-window,month-1).to_numpy()
                        ev=(t.t==month).to_numpy()
                        assert fit.sum()==count*window and not (fit&ev).any()
                        assert not (fit&~dev).any()
                        used|=fit
                        y=np.log1p(t.loc[fit,b.Y].to_numpy())
                        preds["target_mean"][ev]=t.loc[fit,b.Y].mean()
                        x=t.loc[fit,FEATURES];ex=t.loc[ev,FEATURES]
                        lm=b.ridge().fit(x,y)
                        preds["target_log_ridge"][ev]=inverse_prediction(lm,x,ex,y)
                        tm=b.estimator(target=True).fit(x,y)
                        preds["target_log_lgbm"][ev]=inverse_prediction(tm,x,ex,y)
                        for name,p in sp.items():
                            xx=p[fit,None];ee=p[ev,None]
                            m=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(xx,y)
                            preds[name][ev]=inverse_prediction(m,xx,ee,y)
                            coef.append({"direction":direction,"seed":seed,"sentinels":count,"window":window,
                                         "month_index":month,"model":name,"standardized_slope":float(m[-1].coef_[0])})
                    for name,p in preds.items():assert np.isfinite(p[t.year>=2019]).all(),name
                    candidates={k:preds[k][stages["discovery"]] for k in sp}
                    local={k:preds[k][stages["discovery"]] for k in names if k.startswith("target_")}
                    chosen,searchrows=search(df,candidates,local)
                    settings={"direction":direction,"seed":seed,"sentinel_cities":count,"window_months":window,
                              "labels_per_update":count*window,"cumulative_fit_labels":int(used.sum()),"gate_labels":len(df)}
                    lock={**settings,**chosen};locks.append(lock)
                    b.dump(OUT/f"lock_{direction}_{seed}_{count}_{window}.json",lock)
                    grid.extend([{**settings,**r,"rule":json.dumps(r["rule"])} for r in searchrows])
                    gate=weather_valid(t)&b.apply_rule(t,chosen["rule"])
                    candidate=preds[chosen["model"]];reference=preds[chosen["baseline"]]
                    for stage,idx in stages.items():
                        total=t.loc[idx]
                        for name,p in preds.items():
                            selected=idx&weather_valid(t)
                            allmodels.append({**settings,"stage":stage,"model":name,
                                              **b.statistics(t.loc[selected],p[selected],total)})
                        sel=idx&gate
                        a=b.statistics(t.loc[sel],candidate[sel],total)
                        r=b.statistics(t.loc[sel],reference[sel],total)
                        ci=b.bootstrap(t.loc[sel],candidate[sel],reference[sel],seed) if sel.any() else {}
                        gain=1-a["rmse"]/r["rmse"] if r["rmse"]>0 else np.nan
                        passed=bool(chosen["discovery_pass"] and gain>=.05 and a["coverage"]>=.25 and a["fua_coverage"]>=.25
                                    and a["fuas"]>=5 and a["months"]>=12 and ci.get("delta_rmse_ci_low",-np.inf)>0)
                        metrics.append({**settings,"stage":stage,"model":chosen["model"],"baseline":chosen["baseline"],
                            "rule":json.dumps(chosen["rule"]),"mandatory_weather_eligibility":True,
                            "learned_conditions":chosen["learned_conditions"],"discovery_pass":chosen["discovery_pass"],
                            "stage_pass":passed,"baseline_rmse":r["rmse"],"relative_improvement":gain,**a,**ci})
                    ev=np.logical_or.reduce(list(stages.values()))
                    detail=t.loc[ev,["eFUA_ID","fua_name","year","month",b.Y]+b.P["gate_features"]].copy()
                    detail["prediction"]=candidate[ev];detail["baseline_prediction"]=reference[ev]
                    detail["accepted"]=gate[ev];detail["development_city"]=dev[ev]
                    detail.to_csv(OUT/f"predictions_{direction}_{seed}_{count}_{window}.csv.gz",index=False,float_format="%.17g")
                    print("anomaly_done",direction,seed,count,window,"model",chosen["model"],"pass",chosen["discovery_pass"],flush=True)
                pd.DataFrame(metrics).to_csv(OUT/"gates.csv",index=False)
                pd.DataFrame(allmodels).to_csv(OUT/"all_models.csv",index=False)
    pd.DataFrame(grid).to_csv(OUT/"rule_search.csv.gz",index=False)
    pd.DataFrame(coef).to_csv(OUT/"coefficients.csv",index=False)
    b.dump(OUT/"locks.json",locks)
    b.dump(OUT/"run_manifest.json",{"status":"completed","elapsed_seconds":time.time()-start,
        "protocol_sha256":b.sha(HERE/"anomaly_protocol.json"),"script_sha256":b.sha(__file__),
        "input_provenance":provenance,"fit_time_violations":0,"held_city_target_fit_violations":0})
    print("ANOMALY_COMPLETE",time.time()-start,flush=True)


if __name__=="__main__":main()
