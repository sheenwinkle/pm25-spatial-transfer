from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import run_study as base

HERE=Path(__file__).resolve().parent
OUT=HERE/"sequential_results"
CFG=json.loads((HERE/"sequential_protocol.json").read_text(encoding="utf-8"))


def main():
    start=time.time()
    OUT.mkdir(exist_ok=True)
    d,features,provenance=joblib.load(HERE/"fua_month_cache.joblib")
    records=[]; ungated=[]; locks=[]; scans=[]; predictions=[]; coefficients=[]
    for direction in ["CHN_to_AUS","AUS_to_CHN"]:
        source,target=direction.split("_to_")
        t=d[(d.country_iso==target)&(d.year>=2017)].copy().reset_index(drop=True)
        oldpred=pd.read_csv(base.OUT/f"predictions_{direction}_5993.csv.gz")
        assert np.array_equal(t[["eFUA_ID","year","month"]].to_numpy(),oldpred[["eFUA_ID","year","month"]].to_numpy())
        t["support_distance"]=oldpred.support_distance.to_numpy()
        sp={k:oldpred[k].to_numpy() for k in ["source_portable","source_weighted"]}
        for seed in base.P["seeds"]:
            manifest=pd.read_csv(base.OUT/f"city_manifest_{direction}.csv")
            dev_ids=manifest.loc[(manifest.seed==seed)&manifest.development_city,"eFUA_ID"].to_numpy()
            dev=t.eFUA_ID.isin(dev_ids).to_numpy()
            stages={"discovery":dev & t.year.between(2019,2020).to_numpy(),
                    "confirmation":dev & t.year.between(2021,2022).to_numpy(),
                    "spatial_temporal_audit":~dev & t.year.between(2023,2024).to_numpy(),
                    "seen_city_temporal_audit":dev & t.year.between(2023,2024).to_numpy()}
            df=t.loc[stages["discovery"]].reset_index(drop=True)
            atoms=base.masks_for_atoms(df)
            for window in CFG["rolling_target_windows_months"]:
                names=["target_mean","target_ridge","target_lgbm"]
                for name in sp:names += [name+"_affine",name+"_augmented_ridge",name+"_residual_ridge"]
                pred={name:np.full(len(t),np.nan) for name in names}
                used=np.zeros(len(t),dtype=bool)
                for month in sorted(t.loc[t.year>=2019,"t"].unique()):
                    fit=dev & t.t.between(month-window,month-1).to_numpy()
                    ev=(t.t==month).to_numpy()
                    assert len(t.loc[fit])==10*window and not (fit & ev).any()
                    assert (t.loc[fit,"t"]<month).all()
                    used |= fit
                    x=t.loc[fit,features]
                    ex=t.loc[ev,features]
                    y=t.loc[fit,base.Y].to_numpy()
                    pred["target_mean"][ev]=y.mean()
                    pred["target_ridge"][ev]=base.positive(base.ridge().fit(x,y).predict(ex))
                    pred["target_lgbm"][ev]=base.positive(base.estimator(target=True).fit(x,y).predict(ex))
                    for name,p in sp.items():
                        affine=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(p[fit,None],y)
                        pred[name+"_affine"][ev]=base.positive(affine.predict(p[ev,None]))
                        ax=x.copy();ax["source_prediction"]=p[fit]
                        aex=ex.copy();aex["source_prediction"]=p[ev]
                        augmented=base.ridge().fit(ax,y)
                        pred[name+"_augmented_ridge"][ev]=base.positive(augmented.predict(aex))
                        residual=base.ridge().fit(x,y-p[fit])
                        pred[name+"_residual_ridge"][ev]=base.positive(p[ev]+residual.predict(ex))
                        coefficients.append({"direction":direction,"seed":seed,"window":window,"month_index":month,
                            "source":name,"affine_slope":float(affine[-1].coef_[0]/affine[0].scale_[0]),
                            "augmented_standardized_source_coefficient":float(augmented[-1].coef_[-1])})
                assert not (used & ~dev).any(), "Held-city target entered fitting"
                for name in pred:assert np.isfinite(pred[name][t.year>=2019]).all()
                local={k:pred[k][stages["discovery"]] for k in names if k.startswith("target_")}
                transfer={k:pred[k][stages["discovery"]] for k in names if not k.startswith("target_")}
                selected,grid=base.gate_search(df,transfer,local,atoms)
                lock={"direction":direction,"seed":seed,"window_months":window,"fit_labels_per_step":10*window,
                      "unique_fit_labels_full_sequence":int(used.sum()),"gate_labels":len(df),**selected}
                locks.append(lock)
                base.dump(OUT/f"lock_{direction}_{seed}_{window}.json",lock)
                for row in grid:
                    scans.append({"direction":direction,"seed":seed,"window_months":window,
                                  **{k:v for k,v in row.items() if k!="rule"},"rule":json.dumps(row["rule"])})
                accepted=base.apply_rule(t,selected["rule"])
                candidate=pred[selected["model"]]
                reference=pred[selected["baseline"]]
                for stage,idx in stages.items():
                    frame=t.loc[idx]
                    for name in pred:
                        ungated.append({"direction":direction,"seed":seed,"window_months":window,
                            "stage":stage,"model":name,**base.statistics(frame,pred[name][idx],frame)})
                    sel=idx & accepted
                    a=base.statistics(t.loc[sel],candidate[sel],frame)
                    b=base.statistics(t.loc[sel],reference[sel],frame)
                    ci=base.bootstrap(t.loc[sel],candidate[sel],reference[sel],seed) if sel.any() else {}
                    gain=1-a["rmse"]/b["rmse"] if b["rmse"]>0 else np.nan
                    passes=bool(selected["discovery_pass"] and gain>=.05 and a["coverage"]>=.25 and a["fuas"]>=5
                                and a["fua_coverage"]>=.25 and a["months"]>=12 and ci.get("delta_rmse_ci_low",-np.inf)>0)
                    records.append({"direction":direction,"seed":seed,"window_months":window,
                        "stage":stage,"model":selected["model"],"baseline":selected["baseline"],
                        "rule":json.dumps(selected["rule"]),"conditions":selected["conditions"],
                        "fit_labels_per_step":10*window,"gate_labels":len(df),
                        "unique_fit_labels_full_sequence":int(used.sum()),
                        "discovery_pass":selected["discovery_pass"],"stage_pass":passes,
                        "baseline_rmse":b["rmse"],"relative_improvement":gain,**a,**ci})
                    detail=t.loc[idx,["eFUA_ID","fua_name","year","month","quarter_id",base.Y]+base.P["gate_features"]].copy()
                    detail["prediction"]=candidate[idx]
                    detail["baseline_prediction"]=reference[idx]
                    detail["accepted"]=accepted[idx]
                    detail["direction"]=direction;detail["seed"]=seed
                    detail["window_months"]=window;detail["stage"]=stage
                    predictions.append(detail)
                print("sequential_done",direction,seed,window,"gate",selected["rule"],
                      "discovery_pass",selected["discovery_pass"],flush=True)
            # Write completed configurations incrementally.
            pd.DataFrame(records).to_csv(OUT/"gates.csv",index=False)
            pd.DataFrame(ungated).to_csv(OUT/"all_models.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/"selected_predictions.csv.gz",index=False)
    pd.DataFrame(scans).to_csv(OUT/"rule_search.csv.gz",index=False)
    pd.DataFrame(coefficients).to_csv(OUT/"source_coefficients.csv",index=False)
    base.dump(OUT/"locks.json",locks)
    base.dump(OUT/"run_manifest.json",{"status":"completed","elapsed_seconds":time.time()-start,
        "protocol_sha256":base.sha(HERE/"sequential_protocol.json"),"script_sha256":base.sha(__file__),
        "base_script_sha256":base.sha(HERE/"run_study.py"),"input_provenance":provenance,
        "fit_time_violations":0,"held_city_target_fit_violations":0})
    print("SEQUENTIAL_COMPLETE",time.time()-start,flush=True)


if __name__=="__main__":main()
