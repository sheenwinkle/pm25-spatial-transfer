"""Matched weather ablations and target-refitted source partition kernels."""
from __future__ import annotations

import itertools
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS","4")
os.environ.setdefault("OPENBLAS_NUM_THREADS","4")
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import KernelCenterer, StandardScaler

from audit_inputs import HERE, PREVIOUS
from collect_monthly_power import VARIABLES, PROCESSED, META, dump, sha
sys.path.insert(0,str(PREVIOUS))
import run_study as b

P=json.loads((HERE/"protocol.json").read_text(encoding="utf-8"))
OUT=HERE/"results"
WEATHER=list(VARIABLES.values())
CONTEXT=["local_season_sin","local_season_cos","log_population_density","distance_to_coast_km",
         "urban_centre_area_share","log_fire_count_100km"]
LOCAL=P["local_comparators"]
CONTROL="shuffled_source_kernel"
CANDIDATES=P["source_candidates"]


def prepare():
    manifest=json.loads((META/"data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="complete"
    assert manifest["fua_monthly_sha256"]==sha(PROCESSED/"fua_monthly_power_1998_2025.csv.gz")
    d,_,old_provenance=joblib.load(PREVIOUS/"fua_month_cache.joblib")
    d=d.copy();d[b.Y]=d[b.Y].astype(float)
    weather=pd.read_csv(PROCESSED/"fua_monthly_power_1998_2025.csv.gz")
    cols=["eFUA_ID","year","month","power_lon","power_lat","power_great_circle_distance_km"]+WEATHER
    d=d.merge(weather[cols],on=["eFUA_ID","year","month"],how="left",validate="one_to_one")
    assert d[WEATHER].notna().all().all()
    annual=pd.read_csv(PROCESSED/"power_grid_annual_1998_2025.csv.gz")
    for kind in ["legacy_annual","updated_annual"]:
        keyyear=d.year.clip(upper=2022) if kind=="legacy_annual" else d.year
        lookup=d[["eFUA_ID","year","month","power_lon","power_lat"]].copy()
        lookup["weather_reference_year"]=keyyear
        a=annual.rename(columns={"year":"weather_reference_year",**{c:kind+"_"+c for c in WEATHER}})
        lookup=lookup.merge(a,on=["power_lon","power_lat","weather_reference_year"],how="left",validate="many_to_one")
        d=d.merge(lookup[["eFUA_ID","year","month"]+[kind+"_"+c for c in WEATHER]],
            on=["eFUA_ID","year","month"],how="left",validate="one_to_one")
    d["weather_grid_id"]=pd.factorize(pd.MultiIndex.from_frame(d[["power_lon","power_lat"]]))[0]
    for new,old in [("temperature_c",WEATHER[0]),("humidity_pct",VARIABLES["RH2M"]),("wind_mps",VARIABLES["WS10M"])]:d[new]=d[old]
    feature_sets={"legacy_annual":["legacy_annual_"+c for c in WEATHER]+CONTEXT,
                  "updated_annual":["updated_annual_"+c for c in WEATHER]+CONTEXT,
                  "monthly_common":WEATHER+CONTEXT,"monthly_with_aviation":WEATHER+CONTEXT+b.SURFACE}
    for name in ["legacy_annual","updated_annual","monthly_common"]:assert d[feature_sets[name][:6]].notna().all().all()
    assert len(d)==467700 and not d.duplicated(["country_iso","eFUA_ID","year","month"]).any()
    assert np.array_equal(d[b.Y].to_numpy(),joblib.load(PREVIOUS/"fua_month_cache.joblib")[0][b.Y].to_numpy())
    provenance={"original":old_provenance,"recovered_weather_manifest_sha256":sha(META/"data_manifest.json"),
        "fua_monthly_sha256":manifest["fua_monthly_sha256"],"feature_sets":feature_sets,"pm25_labels_changed":0}
    dump(OUT/"features_and_provenance.json",provenance)
    joblib.dump((d,feature_sets,provenance),HERE/"modelling_cache.joblib",compress=3)
    return d,feature_sets,provenance


def normalizer():
    return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler())


def ridge():
    return make_pipeline(SimpleImputer(strategy="median",keep_empty_features=True),StandardScaler(),Ridge(alpha=P["ridge_alpha"]))


def target_lgbm():
    return lgb.LGBMRegressor(**P["target_lightgbm"],random_state=5993,n_jobs=4,verbosity=-1)


def leaf_embedding(model,x):
    leaves=model.predict(x,pred_leaf=True).astype(np.int32)
    if leaves.ndim==1:leaves=leaves[:,None]
    n,nt=leaves.shape;width=P["source_lightgbm"]["num_leaves"]
    assert leaves.min()>=0 and leaves.max()<width
    indices=(leaves+np.arange(nt,dtype=np.int32)*width).ravel()
    return sparse.csr_matrix((np.full(n*nt,1/np.sqrt(nt)),indices,np.arange(n+1,dtype=np.int64)*nt),shape=(n,nt*width))


def kernel_predict(train_kernel,test_kernel,y):
    center=KernelCenterer()
    k=center.fit_transform(train_kernel)
    model=KernelRidge(alpha=P["kernel_ridge_alpha"],kernel="precomputed").fit(k,y-y.mean())
    return b.positive(y.mean()+model.predict(center.transform(test_kernel)))


def support(s,t):
    fs=WEATHER+["local_season_sin","local_season_cos"]
    norm=normalizer();sx=norm.fit_transform(s[fs]);tx=norm.transform(t[fs])
    ids=s.weather_grid_id.unique().copy();rng=np.random.default_rng(5993);rng.shuffle(ids)
    reference=s.weather_grid_id.isin(ids[:max(1,int(.8*len(ids)))]).to_numpy()
    ri=np.flatnonzero(reference);ci=np.flatnonzero(~reference)
    ri=rng.choice(ri,min(10000,len(ri)),replace=False);ci=rng.choice(ci,min(2000,len(ci)),replace=False)
    tree=cKDTree(sx[ri]);distance=tree.query(tx,k=1,workers=4)[0]
    threshold=float(np.quantile(tree.query(sx[ci],k=1,workers=4)[0],.95))
    return distance,threshold


def atom_masks(frame):
    atoms=[]
    for feature in P["gate_features"]:
        v=frame[feature].to_numpy(float)
        if np.isfinite(v).sum()<20:continue
        for threshold in np.unique(np.nanquantile(v,[.1,.25,.5,.75,.9])):
            for op in ["<=",">="]:
                m=v<=threshold if op=="<=" else v>=threshold
                if .2<=m.mean()<.99:atoms.append(({"feature":feature,"operator":op,"threshold":float(threshold)},m))
    return atoms


def bootstrap_comparators(frame,pred,comparators,seed,cluster):
    if not len(frame):return {}
    city,_=pd.factorize(frame[cluster]);quarter,_=pd.factorize(frame.quarter_id)
    nc,nt=city.max()+1,quarter.max()+1
    if nc<2 or nt<2:return {}
    count=np.zeros((nc,nt));np.add.at(count,(city,quarter),1)
    rng=np.random.default_rng(seed)
    cw=rng.multinomial(nc,np.full(nc,1/nc),size=1000)
    tw=rng.multinomial(nt,np.full(nt,1/nt),size=1000)
    den=np.sum((cw@count)*tw,axis=1);valid=den>0
    def rmse(v):
        sums=np.zeros_like(count);np.add.at(sums,(city,quarter),(v-frame[b.Y].to_numpy())**2)
        return np.sqrt(np.sum((cw@sums)*tw,axis=1)[valid]/den[valid])
    pa=rmse(pred);out={}
    for name,p in comparators.items():
        lo,hi=np.quantile(rmse(p)-pa,[.025,.975])
        out[name+"_delta_low"]=float(lo);out[name+"_delta_high"]=float(hi)
    return out


def run(d,features,name,direction):
    source,target=direction.split("_to_")
    s=d[d.country_iso.eq(source)&(d.year<=2018)].copy()
    t=d[d.country_iso.eq(target)&(d.year>=2017)].copy().reset_index(drop=True)
    model=lgb.LGBMRegressor(**P["source_lightgbm"],random_state=5993,n_jobs=4,verbosity=-1).fit(s[features],s[b.Y])
    shuffled=np.random.default_rng(5993).permutation(s[b.Y].to_numpy())
    negative=lgb.LGBMRegressor(**P["source_lightgbm"],random_state=5993,n_jobs=4,verbosity=-1).fit(s[features],shuffled)
    raw=b.positive(model.predict(t[features]));embedding=leaf_embedding(model,t[features]);control=leaf_embedding(negative,t[features])
    t["support_distance"],threshold=support(s,t)
    folder=OUT/name/direction;folder.mkdir(parents=True,exist_ok=True)
    joblib.dump({"source":model,"shuffled":negative,"features":features},folder/"source_models.joblib",compress=3)
    dump(folder/"source_diagnostics.json",{"source_rows":len(s),"source_fuas":s.eFUA_ID.nunique(),
        "source_weather_grids":s.weather_grid_id.nunique(),"source_cross_grid_distance_95":threshold,
        "target_support_fraction_2023_2024":float((t.loc[t.year>=2023,"support_distance"]<=threshold).mean()),
        "features":features,"source_only_2023_2024":b.statistics(t.loc[t.year>=2023],raw[t.year>=2023],t.loc[t.year>=2023])})
    print("SOURCE_READY",name,direction,flush=True)
    metrics=[];gates=[];searchrows=[];budget_rows=[]
    manifest=pd.read_csv(PREVIOUS/f"results/city_manifest_{direction}.csv")
    for seed in P["seeds"]:
        ids=manifest.loc[manifest.seed.eq(seed)&manifest.development_city,"eFUA_ID"].to_numpy()
        dev=t.eFUA_ID.isin(ids).to_numpy();devgrids=set(t.loc[dev,"weather_grid_id"])
        stages={"discovery":dev&t.year.between(2019,2020).to_numpy(),
            "confirmation":dev&t.year.between(2021,2022).to_numpy(),
            "spatial_temporal_audit":~dev&t.year.between(2023,2024).to_numpy(),
            "seen_city_temporal_audit":dev&t.year.between(2023,2024).to_numpy(),
            "new_weather_grid_audit":~dev&~t.weather_grid_id.isin(devgrids).to_numpy()&t.year.between(2023,2024).to_numpy()}
        for window in P["windows_months"]:
            setting={"feature_set":name,"direction":direction,"seed":seed,"window_months":window,"labels_per_fit":10*window}
            preds={k:np.full(len(t),np.nan) for k in LOCAL+CANDIDATES+[CONTROL]}
            similarity=np.full(len(t),np.nan);used=np.zeros(len(t),bool)
            for month in sorted(t.loc[t.year>=2019,"t"].unique()):
                fit=dev&t.t.between(month-window,month-1).to_numpy();ev=t.t.eq(month).to_numpy()
                assert fit.sum()==10*window and not (fit&ev).any() and not (fit&~dev).any()
                used|=fit;y=t.loc[fit,b.Y].to_numpy();x=t.loc[fit,features];ex=t.loc[ev,features]
                preds["target_mean"][ev]=y.mean()
                preds["target_ridge"][ev]=b.positive(ridge().fit(x,y).predict(ex))
                preds["target_lgbm"][ev]=b.positive(target_lgbm().fit(x,y).predict(ex))
                norm=normalizer();zx=norm.fit_transform(x);zex=norm.transform(ex)
                gamma=1/len(features)
                preds["target_rbf"][ev]=kernel_predict(rbf_kernel(zx,gamma=gamma),rbf_kernel(zex,zx,gamma=gamma),y)
                preds["source_affine"][ev]=b.positive(ridge().fit(raw[fit,None],y).predict(raw[ev,None]))
                preds["source_augmented_ridge"][ev]=b.positive(ridge().fit(np.c_[x.to_numpy(),raw[fit]],y).predict(np.c_[ex.to_numpy(),raw[ev]]))
                preds["source_residual_lgbm"][ev]=b.positive(raw[ev]+target_lgbm().fit(x,y-raw[fit]).predict(ex))
                for label,em in [("source_partition_kernel",embedding),(CONTROL,control)]:
                    tx=em[fit];tt=(tx@tx.T).toarray();vt=(em[ev]@tx.T).toarray()
                    preds[label][ev]=kernel_predict(tt,vt,y)
                    if label=="source_partition_kernel":similarity[ev]=vt.max(axis=1)
            assert all(np.isfinite(p[t.year>=2019]).all() for p in preds.values())
            t["calibration_similarity"]=similarity
            discovery=stages["discovery"]
            pfit={k:preds[k][discovery] for k in CANDIDATES}
            controls={k:preds[k][discovery] for k in LOCAL+[CONTROL]}
            chosen,grid=b.gate_search(t.loc[discovery],pfit,controls,atom_masks(t.loc[discovery]))
            selection={**setting,**chosen,"cumulative_fitted_labels":int(used.sum()),"gate_labels":int(discovery.sum()),
                       "fit_and_gate_unique_labels":int((used|discovery).sum())}
            budget_rows.append({k:v for k,v in selection.items() if k not in ["rule"]})
            dump(folder/f"lock_{seed}_{window}.json",selection)
            searchrows.extend([{**setting,**r,"rule":json.dumps(r["rule"])} for r in grid])
            accepted=b.apply_rule(t,chosen["rule"]);candidate=preds[chosen["model"]]
            all_eval=np.logical_or.reduce(list(stages.values()))
            for stage,idx in stages.items():
                total=t.loc[idx];sel=idx&accepted;part=t.loc[sel]
                for k,p in preds.items():
                    for scope,mask in [("all",idx),("frozen_rule",sel)]:
                        metrics.append({**setting,"stage":stage,"scope":scope,"model":k,
                            **b.statistics(t.loc[mask],p[mask],total),"weather_grids":t.loc[mask,"weather_grid_id"].nunique()})
                score=b.statistics(part,candidate[sel],total)
                errors={k:b.statistics(part,p[sel],total)["rmse"] for k,p in preds.items() if k in LOCAL+[CONTROL]}
                best=min(errors,key=errors.get)
                gain=1-score["rmse"]/errors[best] if errors[best]>0 else np.nan
                paired={"frozen":preds[chosen["baseline"]][sel],"mean":preds["target_mean"][sel],"shuffled":preds[CONTROL][sel]}
                ci={}
                for cluster,label in [("eFUA_ID","fua"),("weather_grid_id","grid")]:
                    ci.update({label+"_"+k:v for k,v in bootstrap_comparators(part,candidate[sel],paired,seed,cluster).items()})
                enough=score["coverage"]>=.25 and score["fua_coverage"]>=.25 and score["fuas"]>=5 and score["months"]>=12
                lower=[ci.get(c+"_"+k+"_delta_low",np.nan) for c in ["fua","grid"] for k in ["frozen","mean","shuffled"]]
                passed=bool(chosen["discovery_pass"] and enough and gain>=.05 and all(v>0 for v in lower))
                gates.append({**setting,"stage":stage,"selected_model":chosen["model"],"frozen_comparator":chosen["baseline"],
                    "rule":json.dumps(chosen["rule"]),"conditions":chosen["conditions"],"discovery_pass":chosen["discovery_pass"],
                    "stage_pass":passed,"strongest_audit_comparator_descriptive":best,"gain_vs_all_comparators":gain,
                    **score,"weather_grids":part.weather_grid_id.nunique(),**{k+"_rmse":v for k,v in errors.items()},**ci})
            cols=["eFUA_ID","fua_name","year","month","t","quarter_id","weather_grid_id",b.Y]+P["gate_features"]
            detail=t.loc[all_eval,cols].copy();detail["development_city"]=dev[all_eval];detail["accepted"]=accepted[all_eval]
            detail["new_weather_grid"]=~t.loc[all_eval,"weather_grid_id"].isin(devgrids)
            detail["selected_prediction"]=candidate[all_eval]
            for k in LOCAL+[CONTROL]:detail[k]=preds[k][all_eval]
            detail.to_csv(folder/f"predictions_{seed}_{window}.csv.gz",index=False,float_format="%.17g")
            print("CONFIG_DONE",name,direction,seed,window,chosen["model"],"discovery",chosen["discovery_pass"],flush=True)
            pd.DataFrame(gates).to_csv(folder/"gates.csv",index=False)
            pd.DataFrame(metrics).to_csv(folder/"model_metrics.csv",index=False)
    pd.DataFrame(searchrows).to_csv(folder/"rule_search.csv.gz",index=False)
    pd.DataFrame(budget_rows).to_csv(folder/"label_budgets.csv",index=False)
    dump(folder/"run_manifest.json",{"status":"complete","script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.json"),
        "base_helper_sha256":sha(PREVIOUS/"run_study.py"),"data_manifest_sha256":sha(META/"data_manifest.json"),
        "source_rows":len(s),"configurations":len(budget_rows),"held_city_label_fit_violations":0,"future_label_fit_violations":0})


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    d,features,provenance=prepare()
    for name in P["feature_sets"]:
        for direction in ["CHN_to_AUS","AUS_to_CHN"]:
            run(d,features[name],name,direction)
    dump(OUT/"run_manifest.json",{"status":"complete","elapsed_seconds":time.time()-start,
        "script_sha256":sha(__file__),"protocol_sha256":sha(HERE/"protocol.json"),"configurations":96,
        "data_provenance":provenance,"base_helper_sha256":sha(PREVIOUS/"run_study.py")})
    print("MODELS_COMPLETE",time.time()-start,flush=True)


if __name__=="__main__":main()
