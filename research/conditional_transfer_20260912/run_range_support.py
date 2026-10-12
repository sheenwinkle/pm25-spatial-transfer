"""Reject extrapolated source scores without modifying any third-round prediction."""
from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

import run_study as b
import run_anomaly as a

HERE=Path(__file__).resolve().parent
OUT=HERE/"range_support_results"
KEYS=["direction","seed","sentinel_cities","window_months"]


def main():
    start=time.time();OUT.mkdir(exist_ok=True)
    d,_,provenance=joblib.load(HERE/"fua_month_cache.joblib")
    old=pd.read_csv(a.OUT/"gates.csv",float_precision="round_trip")
    rows=[];bounds=[];audit_count=0
    for direction in old.direction.unique():
        source,target=direction.split("_to_")
        t=d[d.country_iso.eq(target)&(d.year>=2017)].copy().reset_index(drop=True)
        models=joblib.load(a.OUT/f"source_{source}_models.joblib")
        scores={name:model.predict(t[a.FEATURES]) for name,model in models["models"].items()}
        manifest=pd.read_csv(b.OUT/f"city_manifest_{direction}.csv")
        for key,g in old[old.direction.eq(direction)].groupby(KEYS):
            _,seed,count,window=key
            settings=dict(zip(KEYS,key))
            ids=manifest.loc[manifest.seed.eq(seed)&manifest.development_city,"eFUA_ID"].to_numpy(copy=True)
            np.random.default_rng(seed).shuffle(ids)
            sentinel=t.eFUA_ID.isin(ids[:count]).to_numpy()
            name=g.model.iloc[0]
            raw=scores[name]
            accepted=np.zeros(len(t),bool)
            low=np.full(len(t),np.nan);high=low.copy();mean=low.copy()
            for month in sorted(t.loc[t.year>=2019,"t"].unique()):
                fit=sentinel&t.t.between(month-window,month-1).to_numpy()
                ev=t.t.eq(month).to_numpy()
                assert fit.sum()==count*window and not (fit&ev).any()
                lo=float(raw[fit].min());hi=float(raw[fit].max())
                accepted[ev]=(raw[ev]>=lo)&(raw[ev]<=hi)
                low[ev]=lo;high[ev]=hi;mean[ev]=float(t.loc[fit,b.Y].mean())
                bounds.append(settings|{"t":month,"model":name,"low":lo,"high":hi,
                    "fit_labels":int(fit.sum()),"fit_end_t":month-1})
            lookup=t[["eFUA_ID","year","month"]].copy()
            lookup["within_calibration_score_range"]=accepted
            lookup["raw_source_score"]=raw;lookup["calibration_score_low"]=low;lookup["calibration_score_high"]=high
            lookup["target_mean_prediction"]=mean
            q=pd.read_csv(a.OUT/f"predictions_{direction}_{seed}_{count}_{window}.csv.gz",float_precision="round_trip")
            q[b.Y]=q[b.Y].astype(d[b.Y].dtype)
            q=q.merge(lookup,on=["eFUA_ID","year","month"],validate="one_to_one")
            q["t"]=q.year*12+q.month;q["quarter_id"]=q.year*4+(q.month-1)//3
            q["accepted_with_range"]=q.accepted&q.within_calibration_score_range
            dev=q.development_city.to_numpy()
            stages={"discovery":dev&q.year.between(2019,2020).to_numpy(),
                    "confirmation":dev&q.year.between(2021,2022).to_numpy(),
                    "spatial_temporal_audit":~dev&q.year.between(2023,2024).to_numpy(),
                    "seen_city_temporal_audit":dev&q.year.between(2023,2024).to_numpy()}
            discovery_ok=False
            for stage,idx in stages.items():
                total=q.loc[idx];part=total.loc[total.accepted_with_range]
                s=b.statistics(part,part.prediction.to_numpy(),total)
                refs={"frozen":part.baseline_prediction.to_numpy(),"mean":part.target_mean_prediction.to_numpy()}
                extra={}
                for label,p in refs.items():
                    other=b.statistics(part,p,total)
                    ci=b.bootstrap(part,part.prediction.to_numpy(),p,seed) if len(part) else {}
                    extra[label+"_rmse"]=other["rmse"]
                    extra[label+"_gain"]=1-s["rmse"]/other["rmse"] if other["rmse"]>0 else np.nan
                    extra[label+"_delta_low"]=ci.get("delta_rmse_ci_low",np.nan)
                    extra[label+"_delta_high"]=ci.get("delta_rmse_ci_high",np.nan)
                support=s["coverage"]>=.25 and s["fua_coverage"]>=.25 and s["fuas"]>=5 and s["months"]>=12
                gain=extra["frozen_gain"]>=.05 and extra["mean_gain"]>=.05
                if stage=="discovery":
                    annual=[]
                    for year in [2019,2020]:
                        z=part[part.year.eq(year)]
                        annual.extend([bool(len(z) and np.mean((z.prediction-z[b.Y])**2)<np.mean((z[c]-z[b.Y])**2))
                            for c in ["baseline_prediction","target_mean_prediction"]])
                    discovery_ok=bool(support and gain and all(annual))
                passed=bool(discovery_ok and support and gain and extra["frozen_delta_low"]>0 and extra["mean_delta_low"]>0)
                rows.append(settings|{"stage":stage,"model":name,"original_rule":g.rule.iloc[0],
                    "discovery_pass":discovery_ok,"stage_pass":passed,
                    "labels_per_update":int(g.labels_per_update.iloc[0]),
                    "original_row_coverage":float(total.accepted.mean()),**s,**extra})
                audit_count+=1
            expected=q.accepted&(q.raw_source_score>=q.calibration_score_low)&(q.raw_source_score<=q.calibration_score_high)
            assert np.array_equal(expected.to_numpy(),q.accepted_with_range.to_numpy())
            q.to_csv(OUT/f"predictions_{direction}_{seed}_{count}_{window}.csv.gz",index=False,float_format="%.17g")
            print("range_done",direction,seed,count,window,"discovery",discovery_ok,flush=True)
        pd.DataFrame(rows).to_csv(OUT/"gates.csv",index=False)
    pd.DataFrame(bounds).to_csv(OUT/"rolling_score_bounds.csv",index=False)
    result=pd.DataFrame(rows)
    status=result.pivot(index=KEYS,columns="stage",values="stage_pass").reset_index()
    status["both_later_pass"]=status.confirmation&status.spatial_temporal_audit
    status.to_csv(OUT/"configuration_status.csv",index=False)
    counts=status.groupby("direction").agg(configurations=("both_later_pass","size"),successful=("both_later_pass","sum"))
    b.dump(OUT/"run_manifest.json",{"status":"completed","elapsed_seconds":time.time()-start,
        "script_sha256":b.sha(__file__),"protocol_sha256":b.sha(HERE/"range_support_protocol.json"),
        "reused_anomaly_manifest_sha256":b.sha(a.OUT/"run_manifest.json"),"input_provenance":provenance,
        "configurations":len(status),"stage_evaluations":audit_count,
        "success_by_direction":counts.reset_index().to_dict("records"),
        "range_predicate_checks":len(status),"fit_time_violations":0})
    print(counts.to_string())
    print("RANGE_COMPLETE",time.time()-start,flush=True)


if __name__=="__main__":main()
