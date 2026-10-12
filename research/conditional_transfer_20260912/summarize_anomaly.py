"""Audit the third exploratory round without selecting or changing its rules."""
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

HERE=Path(__file__).resolve().parent
OUT=HERE/"review"
KEYS=["direction","seed","sentinel_cities","window_months"]


def main():
    frame,_,_=joblib.load(HERE/"fua_month_cache.joblib")
    gates=pd.read_csv(a.OUT/"gates.csv",float_precision="round_trip")
    models=pd.read_csv(a.OUT/"all_models.csv")
    audit=[];weather=[];checks=0;budgets=[]
    for country,part in frame[frame.year>=2019].groupby("country_iso"):
        for lo,hi in [(2019,2020),(2021,2022),(2023,2024)]:
            q=part[part.year.between(lo,hi)]
            w=a.weather_valid(q)
            weather.append({"country":country,"start_year":lo,"end_year":hi,"all_rows":len(q),
                "weather_rows":int(w.sum()),"row_coverage":float(w.mean()),
                "all_fuas":q.eFUA_ID.nunique(),"weather_fuas":q.loc[w,"eFUA_ID"].nunique()})
    for key,rows in gates.groupby(KEYS):
        direction,seed,count,window=key
        target=direction.split("_to_")[1]
        t=frame[frame.country_iso.eq(target)].copy()
        manifest=pd.read_csv(b.OUT/f"city_manifest_{direction}.csv")
        ids=manifest.loc[manifest.seed.eq(seed)&manifest.development_city,"eFUA_ID"].to_numpy(copy=True)
        np.random.default_rng(seed).shuffle(ids)
        sent=t.eFUA_ID.isin(ids[:count])
        history=t.loc[sent,["t",b.Y]].groupby("t")[b.Y].mean()
        # Exact same past-only target labels, with no source prediction or weather fit.
        mean_lookup={month:float(history.loc[month-window:month-1].mean())
                     for month in sorted(t.loc[t.year>=2019,"t"].unique())}
        used=set()
        for month in mean_lookup:
            fit=t.loc[sent&t.t.between(month-window,month-1)]
            assert len(fit)==count*window and fit.t.max()<month
            used.update(zip(fit.eFUA_ID,fit.t))
        discovery=t.loc[t.eFUA_ID.isin(ids)&t.year.between(2019,2020)]
        gate_labels=set(zip(discovery.eFUA_ID,discovery.t))
        assert len(used)==int(rows.cumulative_fit_labels.iloc[0])
        budgets.append(dict(zip(KEYS,key))|{"fit_labels_unique":len(used),"gate_labels":len(gate_labels),
            "fit_and_gate_label_union":len(used|gate_labels),"fit_gate_overlap":len(used&gate_labels)})
        q=pd.read_csv(a.OUT/f"predictions_{direction}_{seed}_{count}_{window}.csv.gz",float_precision="round_trip")
        q[b.Y]=q[b.Y].astype(frame[b.Y].dtype)
        q["t"]=q.year*12+q.month
        q["quarter_id"]=q.year*4+(q.month-1)//3
        assert not q.duplicated(["eFUA_ID","t"]).any()
        dev=q.development_city.to_numpy()
        stages={"discovery":dev&q.year.between(2019,2020).to_numpy(),
                "confirmation":dev&q.year.between(2021,2022).to_numpy(),
                "spatial_temporal_audit":~dev&q.year.between(2023,2024).to_numpy(),
                "seen_city_temporal_audit":dev&q.year.between(2023,2024).to_numpy()}
        for _,r in rows.iterrows():
            part=q.loc[stages[r.stage]].copy()
            accepted=a.weather_valid(part)&b.apply_rule(part,json.loads(r.rule))
            assert np.array_equal(accepted,part.accepted.to_numpy())
            z=part.loc[accepted].copy()
            stats=b.statistics(z,z.prediction.to_numpy(),part)
            ref=b.statistics(z,z.baseline_prediction.to_numpy(),part)
            for k in ["n","fuas","months","coverage","fua_coverage","rmse","mae","r2","bias","fua_macro_rmse"]:
                assert np.isclose(stats[k],r[k],atol=3e-5,rtol=3e-5,equal_nan=True),(key,r.stage,k,stats[k],r[k])
            assert np.isclose(ref["rmse"],r.baseline_rmse,atol=3e-5,rtol=3e-5,equal_nan=True)
            expected=bool(r.discovery_pass and r.relative_improvement>=.05 and r.coverage>=.25
                and r.fua_coverage>=.25 and r.fuas>=5 and r.months>=12 and r.delta_rmse_ci_low>0)
            assert expected==r.stage_pass
            mean=z.t.map(mean_lookup).to_numpy()
            ms=b.statistics(z,mean,part)
            ci=b.bootstrap(z,z.prediction.to_numpy(),mean,seed) if len(z) else {}
            audit.append(r.to_dict()|{"no_source_mean_rmse":ms["rmse"],
                "gain_vs_no_source_mean":1-r.rmse/ms["rmse"] if ms["rmse"]>0 else np.nan,
                "mean_delta_ci_low":ci.get("delta_rmse_ci_low",np.nan),
                "mean_delta_ci_high":ci.get("delta_rmse_ci_high",np.nan)})
            checks+=1
    status=gates.pivot(index=KEYS,columns="stage",values="stage_pass").reset_index()
    status["both_later_pass"]=status.confirmation&status.spatial_temporal_audit
    counts=status.groupby("direction").agg(configurations=("both_later_pass","size"),
        successful_configurations=("both_later_pass","sum"))
    audit=pd.DataFrame(audit)
    status.to_csv(OUT/"anomaly_configuration_status.csv",index=False)
    audit.to_csv(OUT/"anomaly_rules_with_mean_ablation.csv",index=False)
    pd.DataFrame(weather).to_csv(OUT/"monthly_weather_coverage.csv",index=False)
    pd.DataFrame(budgets).to_csv(OUT/"anomaly_label_costs.csv",index=False)
    manifest=json.loads((a.OUT/"run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"]=="completed"
    assert manifest["script_sha256"]==b.sha(HERE/"run_anomaly.py")
    assert manifest["protocol_sha256"]==b.sha(HERE/"anomaly_protocol.json")
    issues=models.loc[~np.isfinite(models.rmse)|~np.isfinite(models.r2)]
    issues.to_csv(OUT/"anomaly_nonfinite_model_metrics.csv",index=False)
    audit[audit.stage.eq("spatial_temporal_audit")&audit.seed.eq(5993)].to_csv(OUT/"anomaly_primary_audit.csv",index=False)
    summary={"stage_metric_and_rule_checks":checks,"status":"verified",
        "configurations":len(status),"success_by_direction":counts.reset_index().to_dict("records"),
        "model_stage_rows":len(models),"nonfinite_model_stage_rows":len(issues),
        "searched_model_rules":len(pd.read_csv(a.OUT/"rule_search.csv.gz")),
        "source_only_labels":"source through 2018", "source_protocol_script_hashes":"matched",
        "warning":"Rules are exploratory; no-source mean comparisons are diagnostics, not a second selection stage."}
    b.dump(OUT/"anomaly_verification_summary.json",summary)
    plot=audit[audit.direction.eq("CHN_to_AUS")&audit.stage.eq("spatial_temporal_audit")]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
    palette={2:"#b44b51",5:"#277f6c",10:"#397ba6"}
    shapes={5993:"o",5994:"s",5995:"^"}
    for k,c in palette.items():
        for seed,m in shapes.items():
            p=plot[plot.sentinel_cities.eq(k)&plot.seed.eq(seed)]
            axes[0].scatter(p.coverage*100,p.rmse/p.no_source_mean_rmse,color=c,marker=m,s=44,alpha=.8)
            axes[1].scatter(p.labels_per_update,p.rmse/p.no_source_mean_rmse,color=c,marker=m,s=44,alpha=.8)
    for ax in axes:
        ax.set_yscale("log");ax.axhline(1,color="#333333",lw=1,ls="--")
        ax.axhline(.95,color="#a0a0a0",lw=.8,ls=":")
        ax.grid(axis="y",alpha=.2);ax.spines[["top","right"]].set_visible(False)
        ax.set_ylabel("RMSE / same-budget target mean (log scale)")
    axes[0].set_xlabel("Accepted FUA-months (%)");axes[0].set_xlim(0,105)
    axes[1].set_xlabel("Labels per rolling fit");axes[1].set_xscale("log")
    handles=[plt.Line2D([],[],color=c,marker="o",linestyle="",label=f"{k} sentinel cities") for k,c in palette.items()]
    axes[1].legend(handles=handles,frameon=False,loc="upper right")
    fig.suptitle("Monthly-weather transfer: China to Australia, other cities 2023-2024\nThree overlapping city partitions; every frozen rule retained",fontsize=12)
    fig.savefig(OUT/"anomaly_coverage_label_cost.png",dpi=180)
    plt.close(fig)
    print(json.dumps(summary,indent=2))
    print(audit[audit.stage.eq("spatial_temporal_audit")&audit.direction.eq("CHN_to_AUS")][KEYS+["model","coverage","rmse","r2","no_source_mean_rmse","gain_vs_no_source_mean","stage_pass"]].to_string(index=False))


if __name__=="__main__":main()
