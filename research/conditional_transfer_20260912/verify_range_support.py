"""Recompute range-rejection metrics and check that original predictions are unchanged."""
from pathlib import Path
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import run_study as b
import run_anomaly as a
import run_range_support as r

HERE=Path(__file__).resolve().parent
REVIEW=HERE/"review"


def main():
    d,_,_=joblib.load(HERE/"fua_month_cache.joblib")
    gates=pd.read_csv(r.OUT/"gates.csv",float_precision="round_trip")
    count=0
    for key,rows in gates.groupby(r.KEYS):
        direction,seed,k,w=key
        name=f"predictions_{direction}_{seed}_{k}_{w}.csv.gz"
        q=pd.read_csv(r.OUT/name,float_precision="round_trip")
        old=pd.read_csv(a.OUT/name,float_precision="round_trip")
        assert np.array_equal(q[["eFUA_ID","year","month"]],old[["eFUA_ID","year","month"]])
        assert np.array_equal(q.prediction.to_numpy(),old.prediction.to_numpy())
        assert np.array_equal(q.baseline_prediction.to_numpy(),old.baseline_prediction.to_numpy())
        q[b.Y]=q[b.Y].astype(d[b.Y].dtype)
        accepted=q.accepted&(q.raw_source_score>=q.calibration_score_low)&(q.raw_source_score<=q.calibration_score_high)
        assert np.array_equal(accepted,q.accepted_with_range)
        dev=q.development_city.to_numpy()
        stages={"discovery":dev&q.year.between(2019,2020).to_numpy(),
                "confirmation":dev&q.year.between(2021,2022).to_numpy(),
                "spatial_temporal_audit":~dev&q.year.between(2023,2024).to_numpy(),
                "seen_city_temporal_audit":dev&q.year.between(2023,2024).to_numpy()}
        for _,row in rows.iterrows():
            total=q.loc[stages[row.stage]];p=total.loc[total.accepted_with_range]
            s=b.statistics(p,p.prediction.to_numpy(),total)
            for name in ["n","fuas","months","coverage","fua_coverage","rmse","mae","r2","bias","fua_macro_rmse"]:
                assert np.isclose(s[name],row[name],atol=3e-5,rtol=3e-5,equal_nan=True),(key,row.stage,name)
            for label,c in [("frozen","baseline_prediction"),("mean","target_mean_prediction")]:
                other=b.statistics(p,p[c].to_numpy(),total)
                assert np.isclose(other["rmse"],row[label+"_rmse"],atol=3e-5,rtol=3e-5,equal_nan=True)
                if len(p):
                    ci=b.bootstrap(p,p.prediction.to_numpy(),p[c].to_numpy(),seed)
                    for side in ["low","high"]:
                        assert np.isclose(ci["delta_rmse_ci_"+side],row[label+"_delta_"+side],atol=3e-5,rtol=3e-5,equal_nan=True)
            count+=1
    bounds=pd.read_csv(r.OUT/"rolling_score_bounds.csv")
    assert (bounds.fit_end_t<bounds.t).all()
    assert (bounds.fit_labels==bounds.sentinel_cities*bounds.window_months).all()
    assert (bounds.low<=bounds.high).all()
    manifest=json.loads((r.OUT/"run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["script_sha256"]==b.sha(HERE/"run_range_support.py")
    assert manifest["protocol_sha256"]==b.sha(HERE/"range_support_protocol.json")
    assert manifest["reused_anomaly_manifest_sha256"]==b.sha(a.OUT/"run_manifest.json")
    summary={"status":"passed","stage_metric_and_ci_checks":count,"unchanged_prediction_configurations":72,
             "rolling_range_time_and_size_checks":len(bounds),"protocol_and_script_hashes":"matched"}
    b.dump(REVIEW/"range_support_verification.json",summary)
    gates[gates.stage.eq("spatial_temporal_audit")&gates.seed.eq(5993)].to_csv(REVIEW/"range_support_primary_audit.csv",index=False)
    p=gates[gates.direction.eq("CHN_to_AUS")&gates.stage.eq("spatial_temporal_audit")]
    fig,ax=plt.subplots(figsize=(9,5),layout="constrained")
    colors={2:"#b44b51",5:"#277f6c",10:"#397ba6"}
    for k,c in colors.items():
        z=p[p.sentinel_cities.eq(k)]
        ax.scatter(z.coverage*100,z.mean_gain*100,color=c,s=48,label=f"{k} sentinel cities",alpha=.85)
    ax.axhline(0,color="#444444",lw=1)
    ax.axhline(5,color="#2b805d",ls="--",lw=1,label="5% exploratory gain threshold")
    ax.axvline(25,color="#777777",ls=":",lw=1,label="25% row-coverage threshold")
    ax.set_xlim(0,102);ax.set_ylim(min(-8,p.mean_gain.min()*100-1),7)
    ax.set_xlabel("Accepted FUA-months (% of all other-city observations)")
    ax.set_ylabel("RMSE reduction versus same-budget local mean (%)")
    ax.set_title("Adding a recent-calibration score interval\nChina to Australia, other cities 2023-2024",fontsize=12)
    ax.spines[["top","right"]].set_visible(False);ax.grid(axis="y",alpha=.18)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.18),fontsize=8,frameon=False,ncols=3)
    fig.savefig(REVIEW/"range_support_gain_coverage.png",dpi=180);plt.close(fig)
    print(json.dumps(summary,indent=2))
    print("max_australia_mean_gain",float(p.mean_gain.max()))


if __name__=="__main__":main()
