"""Retrospective, budget-aware discovery of simple transfer applicability rules."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"
P = json.loads((HERE / "protocol.json").read_text(encoding="utf-8"))
Y = "pm25_mean_ug_m3"
SURFACE = [
    "surface_temperature_mean_c", "surface_relative_humidity_mean_pct",
    "surface_wind_mean_mps", "surface_dry_visibility_harmonic_mean_km",
    "surface_visibility_harmonic_mean_km", "surface_stagnant_day_fraction",
    "surface_precipitation_sum_mm", "surface_sea_level_pressure_mean_hpa",
    "surface_dry_visibility_valid_days", "surface_source_available",
]
POWER = ["power_t2m_c", "power_rh2m_percent", "power_ws10m_m_s",
         "power_prectotcorr_mm_day", "power_elevation_m"]
EMISSIONS = [f"edgar_{s}_emissions_tonnes_gridcell" for s in ["pm25", "nox", "so2", "nh3", "nmvoc"]]
GEOGRAPHIC = ["fua_centroid_lon", "fua_centroid_lat", "year", "month"]
BASE = ["population_density_2015_per_km2", "fua_population_2015",
        "fua_area_km2", "urban_centre_area_share", "distance_to_coast_km"]
UPPER = ["upper_lower_tropospheric_stability_850_1000_c",
         "upper_ventilation_proxy_1000_850_m2_s", "igra_source_available"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2,
                                   default=lambda x: x.item() if isinstance(x, np.generic) else str(x)), encoding="utf-8")


def prepare(root):
    cache = HERE / "fua_month_cache.joblib"
    panel = root / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
    firepath = root / "data/processed/dynamic_sources/fua_month_modis_fire_2000_2025.csv.gz"
    provenance = {"panel": str(panel), "panel_sha256": sha(panel), "fire_sha256": sha(firepath)}
    if cache.exists():
        cached = joblib.load(cache)
        if cached[2] == provenance:
            return cached[0], cached[1], provenance
    numeric = list(dict.fromkeys([Y] + SURFACE + POWER + EMISSIONS + BASE + UPPER + GEOGRAPHIC))
    keys = ["country_iso", "eFUA_ID", "year", "month"]
    cols = list(dict.fromkeys(["cell_id", "fua_name"] + keys + numeric))
    raw = pd.read_csv(panel, usecols=cols, dtype={c: "float32" for c in numeric if c not in keys})
    assert not raw.duplicated(["cell_id", "year", "month"]).any(), "Duplicate grid-months"
    assert raw[Y].notna().all()
    meta = raw.groupby(["country_iso", "eFUA_ID"], observed=True)["fua_name"].first()
    d = raw.groupby(keys, observed=True)[[c for c in numeric if c not in keys]].mean().reset_index()
    d = d.merge(meta.reset_index(), on=["country_iso", "eFUA_ID"], validate="many_to_one")
    support = raw.groupby("country_iso").agg(grid_rows=(Y,"size"), cells=("cell_id","nunique"), fuas=("eFUA_ID","nunique")).reset_index()
    del raw
    f = pd.read_csv(firepath, usecols=["eFUA_ID", "year", "month", "fire_detection_count_100km", "fire_source_available"])
    d = d.merge(f, on=["eFUA_ID", "year", "month"], how="left", validate="one_to_one")
    d["log_fire_count_100km"] = np.log1p(d["fire_detection_count_100km"].clip(lower=0))
    d["log_population_density"] = np.log1p(d["population_density_2015_per_km2"].clip(lower=0))
    d["log_population"] = np.log1p(d["fua_population_2015"].clip(lower=0))
    d["log_fua_area"] = np.log1p(d["fua_area_km2"].clip(lower=0))
    for c in EMISSIONS:
        d["log_" + c] = np.log1p(d[c].clip(lower=0))
    # Encode the seasonal phase, offsetting the southern hemisphere by six months.
    season = (d["month"] - 1 + 6 * (d["fua_centroid_lat"] < 0)) % 12
    d["local_season_sin"] = np.sin(2 * np.pi * season / 12)
    d["local_season_cos"] = np.cos(2 * np.pi * season / 12)
    for name, source in [("temperature_c", SURFACE[0]), ("humidity_pct", SURFACE[1]),
                         ("wind_mps", SURFACE[2]), ("dry_visibility_km", SURFACE[3])]:
        d[name] = d[source]
    # Physical gate variables remain missing when monthly reports are missing.
    d["t"] = d["year"] * 12 + d["month"]
    d["quarter_id"] = d["year"] * 4 + (d["month"] - 1) // 3
    features = POWER + SURFACE + UPPER + ["log_" + c for c in EMISSIONS] + [
        "urban_centre_area_share", "distance_to_coast_km", "log_population_density",
        "log_population", "log_fua_area", "log_fire_count_100km", "fire_source_available",
        "local_season_sin", "local_season_cos"]
    d.replace([np.inf, -np.inf], np.nan, inplace=True)
    assert not d.duplicated(keys).any()
    assert d.groupby(["country_iso", "eFUA_ID"]).size().eq(300).all()
    support.to_csv(OUT / "original_support.csv", index=False)
    dump(OUT / "features.json", {"portable": features, "geographic_additions": GEOGRAPHIC,
                                 "note": "POWER is annual context, not monthly weather. FUA means are not area-weighted exposure estimates."})
    joblib.dump((d, features, provenance), cache, compress=3)
    return d, features, provenance


def estimator(target=False):
    return lgb.LGBMRegressor(**P["target_lightgbm" if target else "source_lightgbm"],
                             random_state=5993, n_jobs=4, verbosity=-1)


def ridge():
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), Ridge(alpha=P["ridge_alpha"]))


def positive(x):
    return np.maximum(np.asarray(x, dtype=float), 0)


def source_predictions(d, features, source, target):
    s = d[(d.country_iso == source) & (d.year <= P["source_train_end"])].copy()
    t = d[(d.country_iso == target) & (d.year >= 2017)].copy().reset_index(drop=True)
    # All pre-2019 target covariates are allowed for an explicitly unsupervised weighting step.
    u = d[(d.country_iso == target) & (d.year <= 2018)].copy()
    pred = {}
    models = {}
    for name, fs in [("source_geo", features + GEOGRAPHIC), ("source_portable", features)]:
        model = estimator().fit(s[fs], s[Y])
        pred[name] = positive(model.predict(t[fs]))
        models[name] = model
    rng = np.random.default_rng(5993)
    n = min(5000, len(s), len(u))
    si = rng.choice(len(s), n, replace=False)
    ui = rng.choice(len(u), n, replace=False)
    domain_x = pd.concat([s.iloc[si][features], u.iloc[ui][features]], ignore_index=True)
    domain_y = np.r_[np.zeros(n), np.ones(n)]
    train, check = train_test_split(np.arange(2*n), test_size=.25, random_state=5993, stratify=domain_y)
    clf = lgb.LGBMClassifier(n_estimators=120, num_leaves=7, min_child_samples=40,
                            learning_rate=.04, random_state=5993, n_jobs=4, verbosity=-1)
    clf.fit(domain_x.iloc[train], domain_y[train])
    auc = roc_auc_score(domain_y[check], clf.predict_proba(domain_x.iloc[check])[:,1])
    clf.fit(domain_x, domain_y)
    prob = np.clip(clf.predict_proba(s[features])[:,1], 1e-4, 1-1e-4)
    weights = np.clip(prob/(1-prob), .05, 20)
    wmodel = estimator().fit(s[features], s[Y], sample_weight=weights)
    pred["source_weighted"] = positive(wmodel.predict(t[features]))
    models["source_weighted"] = wmodel
    # Cross-FUA support proxy calibrated only on source covariates, no target labels.
    distfeatures = POWER + ["log_" + c for c in EMISSIONS] + [
        "log_population_density", "distance_to_coast_km", "local_season_sin", "local_season_cos"]
    norm = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler())
    sx = norm.fit_transform(s[distfeatures])
    tx = norm.transform(t[distfeatures])
    ids = np.array(sorted(s.eFUA_ID.unique()))
    rng.shuffle(ids)
    ref_ids = ids[:max(1, int(.8*len(ids)))]
    refidx = np.flatnonzero(s.eFUA_ID.isin(ref_ids))
    checkidx = np.flatnonzero(~s.eFUA_ID.isin(ref_ids))
    refidx = rng.choice(refidx, min(len(refidx), 5000), replace=False)
    checkidx = rng.choice(checkidx, min(len(checkidx), 1500), replace=False)
    tree = cKDTree(sx[refidx])
    q = tree.query(sx[checkidx], k=1, workers=4)[0]
    threshold = float(np.quantile(q, .95))
    t["support_distance"] = tree.query(tx, k=1, workers=4)[0]
    pred["source_mean"] = np.full(len(t), float(s[Y].mean()))
    summary = {"direction": source+"_to_"+target, "source_train_rows": len(s),
               "source_train_fuas": s.eFUA_ID.nunique(), "target_pre2019_unlabelled_rows": len(u),
               "domain_classifier_row_holdout_auc_descriptive": auc,
               "weight_effective_sample_size": weights.sum()**2/np.sum(weights**2),
               "support_distance_source_95": threshold,
               "target_2023_2024_support_fraction": float((t.loc[t.year>=2023,"support_distance"]<=threshold).mean())}
    joblib.dump({"models":models,"normalizer":norm,"source_reference":sx[refidx],
                 "distance_features":distfeatures,"threshold":threshold}, OUT/(source+"_to_"+target+"_source_models.joblib"), compress=3)
    print("source_done", summary, flush=True)
    return t, pred, summary


def statistics(frame, yhat, total):
    if not len(frame):
        return {"n":0,"fuas":0,"months":0,"coverage":0.,"fua_coverage":0.,
                "rmse":np.nan,"mae":np.nan,"r2":np.nan,"bias":np.nan,"fua_macro_rmse":np.nan}
    residual = yhat - frame[Y].to_numpy()
    var = np.sum((frame[Y].to_numpy()-frame[Y].mean())**2)
    return {"n":len(frame), "fuas":frame.eFUA_ID.nunique(), "months":frame.t.nunique(),
            "coverage":len(frame)/len(total), "fua_coverage":frame.eFUA_ID.nunique()/total.eFUA_ID.nunique(),
            "rmse":float(np.sqrt(np.mean(residual**2))), "mae":float(np.mean(abs(residual))),
            "r2":float(1-np.sum(residual**2)/var) if var>0 else np.nan,
            "bias":float(np.mean(residual)),
            "fua_macro_rmse":float(pd.DataFrame({"fua":frame.eFUA_ID,"se":residual**2}).groupby("fua").se.mean().pow(.5).mean())}


def bootstrap(frame, a, b, seed):
    if frame.eFUA_ID.nunique()<2 or frame.quarter_id.nunique()<2:
        return {"delta_rmse_ci_low":np.nan,"delta_rmse_ci_high":np.nan}
    city, _ = pd.factorize(frame.eFUA_ID)
    timeid, _ = pd.factorize(frame.quarter_id)
    nc, nt = city.max()+1, timeid.max()+1
    count = np.zeros((nc,nt))
    sa, sb = np.zeros_like(count), np.zeros_like(count)
    np.add.at(count,(city,timeid),1)
    y = frame[Y].to_numpy()
    np.add.at(sa,(city,timeid),(a-y)**2)
    np.add.at(sb,(city,timeid),(b-y)**2)
    rng = np.random.default_rng(seed)
    cw = rng.multinomial(nc,np.full(nc,1/nc),size=1000)
    tw = rng.multinomial(nt,np.full(nt,1/nt),size=1000)
    den = np.sum((cw@count)*tw,axis=1)
    ea = np.sum((cw@sa)*tw,axis=1)
    eb = np.sum((cw@sb)*tw,axis=1)
    valid = den>0
    delta = np.sqrt(eb[valid]/den[valid])-np.sqrt(ea[valid]/den[valid])
    lo,hi=np.quantile(delta,[.025,.975])
    return {"delta_rmse_ci_low":float(lo),"delta_rmse_ci_high":float(hi)}


def masks_for_atoms(frame):
    atoms = []
    for feature in P["gate_features"]:
        v=frame[feature].to_numpy(float)
        if np.isfinite(v).sum()<20:
            continue
        for threshold in np.unique(np.nanquantile(v,P["gate_quantiles_on_discovery_predictors"])):
            for op in ["<=",">="]:
                mask=(v<=threshold) if op=="<=" else (v>=threshold)
                if .2<=mask.mean()<.99:
                    atoms.append(({"feature":feature,"operator":op,"threshold":float(threshold)},mask))
    return atoms


def apply_rule(frame, rule):
    accepted=np.ones(len(frame),dtype=bool)
    for node in rule:
        v=frame[node["feature"]].to_numpy(float)
        accepted &= (v<=node["threshold"]) if node["operator"]=="<=" else (v>=node["threshold"])
    return accepted


def gate_search(frame, candidates, locals_, atoms):
    y=frame[Y].to_numpy()
    years=frame.year.to_numpy()
    records=[]
    def evaluate(model, rule, mask):
        if mask.mean()<P["minimum_row_coverage"] or frame.loc[mask,"eFUA_ID"].nunique()<5 or frame.loc[mask,"t"].nunique()<12:
            return None
        e=np.sqrt(np.mean((candidates[model][mask]-y[mask])**2))
        local_errors={k:np.sqrt(np.mean((v[mask]-y[mask])**2)) for k,v in locals_.items()}
        baseline=min(local_errors,key=local_errors.get)
        eb=local_errors[baseline]
        gain=1-e/max(eb,1e-12)
        annual=[]
        for year in np.unique(years):
            m=mask & (years==year)
            annual.append(float(np.sqrt(np.mean((locals_[baseline][m]-y[m])**2))-np.sqrt(np.mean((candidates[model][m]-y[m])**2))) if m.any() else -np.inf)
        rec={"model":model,"baseline":baseline,"rule":rule,"conditions":len(rule),
             "coverage":float(mask.mean()),"rmse":float(e),"baseline_rmse":float(eb),
             "relative_improvement":float(gain),"min_year_improvement":float(min(annual)),
             "discovery_pass":bool(gain>=.05 and min(annual)>0)}
        records.append(rec)
        return rec
    for model in candidates:
        evaluate(model,[],np.ones(len(frame),dtype=bool))
        scored=[]
        for i,(atom,mask) in enumerate(atoms):
            rec=evaluate(model,[atom],mask)
            if rec:
                scored.append((rec["relative_improvement"],i))
        top=[i for _,i in sorted(scored,reverse=True)[:12]]
        for i,j in itertools.combinations(top,2):
            ai,mi=atoms[i];aj,mj=atoms[j]
            evaluate(model,[ai,aj],mi & mj)
    passes=[r for r in records if r["discovery_pass"]]
    if passes:
        selected=sorted(passes,key=lambda r:(r["conditions"],-r["coverage"],-r["relative_improvement"]))[0]
    else:
        selected=max((r for r in records if r["conditions"]==0),key=lambda r:r["relative_improvement"])
    return selected,records


def run_direction(d, features, source, target):
    direction=source+"_to_"+target
    t, source_pred, source_summary=source_predictions(d,features,source,target)
    all_metrics=[]; gate_metrics=[]; scans=[]; locks=[]; manifests=[]; curves=[]
    for seed in P["seeds"]:
        ids=np.array(sorted(t.eFUA_ID.unique()))
        rng=np.random.default_rng(seed)
        dev_ids=np.sort(rng.choice(ids,10,replace=False))
        dev=t.eFUA_ID.isin(dev_ids).to_numpy()
        stages={"discovery":dev & t.year.between(2019,2020).to_numpy(),
                "confirmation":dev & t.year.between(2021,2022).to_numpy(),
                "spatial_temporal_audit":~dev & t.year.between(2023,2024).to_numpy(),
                "seen_city_temporal_audit":dev & t.year.between(2023,2024).to_numpy()}
        df=t.loc[stages["discovery"]].reset_index(drop=True)
        atoms=masks_for_atoms(df)
        manifest=t[["eFUA_ID","fua_name"]].drop_duplicates().assign(direction=direction,seed=seed)
        manifest["development_city"]=manifest.eFUA_ID.isin(dev_ids)
        manifests.append(manifest)
        # Fixed, source-only support gate: no target-outcome-derived threshold.
        for stage,idx in stages.items():
            base=t.loc[idx]
            for name in source_pred:
                all_metrics.append({"direction":direction,"seed":seed,"budget_months":0,"fit_labels":0,
                                    "stage":stage,"model":name,**statistics(base,source_pred[name][idx],base)})
            m=t.support_distance.to_numpy()<=source_summary["support_distance_source_95"]
            sel=idx & m
            gate_metrics.append({"direction":direction,"seed":seed,"budget_months":0,"fit_labels":0,
                "gate_labels":0,"stage":stage,"model":"source_portable","baseline":"source_mean",
                "gate_kind":"source_only_support_gate","rule":json.dumps([{ "feature":"support_distance", "operator":"<=", "threshold":source_summary["support_distance_source_95"]}]),
                **statistics(t.loc[sel],source_pred["source_portable"][sel],base)})
        seed_predictions=t[["eFUA_ID","fua_name","year","month",Y]+P["gate_features"]].copy()
        seed_predictions["development_city"]=dev
        for name,v in source_pred.items():seed_predictions[name]=v
        for budget in P["target_fitting_months"]:
            fit=dev & t.t.between(2018*12+12-budget+1,2018*12+12).to_numpy()
            cal=t.loc[fit]
            assert len(cal)==10*budget
            assert not (fit & np.logical_or.reduce(list(stages.values()))).any()
            local_predictions={"target_mean":np.full(len(t),cal[Y].mean())}
            local_predictions["target_ridge"]=positive(ridge().fit(cal[features],cal[Y]).predict(t[features]))
            local_predictions["target_lgbm"]=positive(estimator(target=True).fit(cal[features],cal[Y]).predict(t[features]))
            transfer_predictions={k:v for k,v in source_pred.items() if k!="source_mean"}
            artifacts={}
            for srcname in ["source_portable","source_weighted"]:
                sp=source_pred[srcname]
                bias=float(np.mean(cal[Y].to_numpy()-sp[fit]))
                transfer_predictions[srcname+"_bias"]=positive(sp+bias)
                # Only the small target fitting set supplies regression labels here.
                affine=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(sp[fit,None],cal[Y])
                transfer_predictions[srcname+"_affine"]=positive(affine.predict(sp[:,None]))
                residual=ridge().fit(cal[features],cal[Y].to_numpy()-sp[fit])
                transfer_predictions[srcname+"_residual_ridge"]=positive(sp+residual.predict(t[features]))
                artifacts[srcname]={"bias":bias,"affine":affine,"residual":residual}
            candidates={k:v[stages["discovery"]] for k,v in transfer_predictions.items()}
            localdev={k:v[stages["discovery"]] for k,v in local_predictions.items()}
            selected,grid=gate_search(df,candidates,localdev,atoms)
            lock={"direction":direction,"seed":seed,"budget_months":budget,"fit_labels":len(cal),
                  "gate_labels":len(df),"discovery_fuas":dev_ids.tolist(),**selected}
            locks.append(lock)
            # Persist before reading any later outcome to score this selected rule.
            dump(OUT/f"lock_{direction}_{seed}_{budget}.json",lock)
            scans.extend([{**{k:v for k,v in r.items() if k!="rule"},"rule":json.dumps(r["rule"]),
                           "direction":direction,"seed":seed,"budget_months":budget} for r in grid])
            candidate=transfer_predictions[selected["model"]]
            reference=local_predictions[selected["baseline"]]
            gate=apply_rule(t,selected["rule"])
            seed_predictions[f"b{budget}_selected_prediction"]=candidate
            seed_predictions[f"b{budget}_baseline_prediction"]=reference
            seed_predictions[f"b{budget}_accepted"]=gate
            full_preds={**local_predictions,**transfer_predictions}
            for stage,idx in stages.items():
                base=t.loc[idx]
                for name,pred in full_preds.items():
                    all_metrics.append({"direction":direction,"seed":seed,"budget_months":budget,
                                        "fit_labels":len(cal),"stage":stage,"model":name,
                                        **statistics(base,pred[idx],base)})
                sel=idx & gate
                stats=statistics(t.loc[sel],candidate[sel],base)
                refstats=statistics(t.loc[sel],reference[sel],base)
                intervals=bootstrap(t.loc[sel],candidate[sel],reference[sel],seed) if sel.any() else {}
                gain=1-stats["rmse"]/refstats["rmse"] if refstats["rmse"]>0 else np.nan
                passes=bool(selected["discovery_pass"] and gain>=.05 and stats["coverage"]>=.25
                            and stats["fuas"]>=5 and stats["fua_coverage"]>=.25 and stats["months"]>=12
                            and intervals.get("delta_rmse_ci_low",-np.inf)>0)
                gate_metrics.append({"direction":direction,"seed":seed,"budget_months":budget,
                    "fit_labels":len(cal),"gate_labels":len(df),"stage":stage,
                    "model":selected["model"],"baseline":selected["baseline"],
                    "gate_kind":"target_developed_rule","rule":json.dumps(selected["rule"]),
                    "discovery_pass":selected["discovery_pass"],"stage_pass":passes,
                    "baseline_rmse":refstats["rmse"],"relative_improvement":gain,**stats,**intervals})
                for feature in ["support_distance","humidity_pct","wind_mps","distance_to_coast_km"]:
                    v=df[feature].to_numpy(float)
                    if np.isfinite(v).sum()<20:continue
                    for quantile in [.1,.25,.5,.75,.9,1.]:
                        threshold=float(np.nanquantile(v,quantile))
                        curve=idx & (t[feature].to_numpy()<=threshold)
                        cs=statistics(t.loc[curve],candidate[curve],base)
                        curves.append({"direction":direction,"seed":seed,"budget_months":budget,
                            "stage":stage,"feature":feature,"discovery_quantile":quantile,
                            "threshold":threshold,**cs})
            joblib.dump(artifacts,OUT/f"adapters_{direction}_{seed}_{budget}.joblib",compress=3)
            print("budget_done",direction,seed,budget,"rule",selected["rule"],
                  "discovery_pass",selected["discovery_pass"],flush=True)
        seed_predictions.to_csv(OUT/f"predictions_{direction}_{seed}.csv.gz",index=False)
    pd.DataFrame(all_metrics).to_csv(OUT/f"metrics_{direction}.csv",index=False)
    pd.DataFrame(gate_metrics).to_csv(OUT/f"gates_{direction}.csv",index=False)
    pd.DataFrame(scans).to_csv(OUT/f"rule_search_{direction}.csv.gz",index=False)
    pd.DataFrame(curves).to_csv(OUT/f"risk_coverage_{direction}.csv",index=False)
    pd.concat(manifests).to_csv(OUT/f"city_manifest_{direction}.csv",index=False)
    dump(OUT/f"locks_{direction}.json",locks)
    return source_summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--reference-root",type=Path,required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    start=time.time()
    d,features,provenance=prepare(args.reference_root)
    print("prepared",len(d),"FUA-months",len(features),"portable features",flush=True)
    dataset=d.groupby("country_iso").agg(rows=(Y,"size"),fuas=("eFUA_ID","nunique"),
                                         target_mean=(Y,"mean"),target_sd=(Y,"std")).reset_index()
    dataset.to_csv(OUT/"fua_support.csv",index=False)
    summaries=[]
    for source,target in [("CHN","AUS"),("AUS","CHN")]:
        summaries.append(run_direction(d,features,source,target))
    pd.DataFrame(summaries).to_csv(OUT/"source_diagnostics.csv",index=False)
    dump(OUT/"run_manifest.json",{"protocol_sha256":sha(HERE/"protocol.json"),"script_sha256":sha(__file__),
         "input_provenance":provenance,"elapsed_seconds":time.time()-start,
         "seeds":P["seeds"],"status":"completed", "label":"retrospective exploratory"})
    print("COMPLETE",time.time()-start,flush=True)


if __name__=="__main__":
    main()
