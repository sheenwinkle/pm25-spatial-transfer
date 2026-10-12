from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
OUT=HERE/"review"
OUT.mkdir(exist_ok=True)
S=pd.concat([pd.read_csv(p) for p in (HERE/"results").glob("gates_*.csv")],ignore_index=True)
S=S[S.gate_kind.eq("target_developed_rule")].copy()
Q=pd.read_csv(HERE/"sequential_results/gates.csv")
S["experiment"]="Static calibration";Q["experiment"]="Rolling calibration"
S["window_months"]=S.budget_months
G=pd.concat([S,Q],ignore_index=True)
keep=["experiment","direction","seed","window_months","stage","model","baseline","rule",
      "n","fuas","coverage","fua_coverage","rmse","mae","r2","baseline_rmse",
      "relative_improvement","delta_rmse_ci_low","delta_rmse_ci_high","discovery_pass","stage_pass"]
G[keep].to_csv(OUT/"all_frozen_rule_results.csv",index=False)
status=G[G.stage.isin(["confirmation","spatial_temporal_audit"])].pivot(
    index=["experiment","direction","seed","window_months"],columns="stage",values="stage_pass")
status["both_pass"]=status.fillna(False).all(axis=1)
status.to_csv(OUT/"configuration_pass_status.csv")
counts=status.groupby(["experiment","direction"]).agg(
    configurations=("both_pass","size"),both_pass=("both_pass","sum"))
counts.to_csv(OUT/"success_counts.csv")

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig,axes=plt.subplots(1,2,figsize=(12,4.9),layout="constrained")
for ax,direction in zip(axes,["CHN_to_AUS","AUS_to_CHN"]):
    for experiment,color,marker in [("Static calibration","#bd5544","o"),("Rolling calibration","#237b7c","s")]:
        rows=G[(G.direction==direction)&(G.stage=="spatial_temporal_audit")&(G.experiment==experiment)]
        ax.scatter(rows.coverage*100,rows.rmse/rows.baseline_rmse,c=color,marker=marker,alpha=.8,s=48,label=experiment)
    ax.axhline(1,color="#333333",lw=1,label="Same-budget target-only baseline")
    ax.axhline(.95,color="#767676",lw=1,ls=":")
    ax.axvline(25,color="#767676",lw=1,ls=":")
    ax.set(yscale="log",xlabel="Retained target FUA-months (%)",ylabel="RMSE / frozen local-baseline RMSE",
           title=direction.replace("_to_"," to "),xlim=(0,105))
    ax.grid(axis="y",alpha=.15)
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,fontsize=8,loc="outside lower center",ncol=3)
fig.suptitle("No configuration passed both later validation stages\n2023-2024 other-city audit; three city partitions, five budgets",fontsize=12)
fig.savefig(OUT/"condition_coverage_audit.png",dpi=180)
plt.close(fig)

models=pd.read_csv(HERE/"sequential_results/all_models.csv")
names={"source_weighted_affine":("Source + affine calibration","#237b7c"),
       "target_mean":("Local rolling mean (no source)","#555555"),
       "target_ridge":("Local Ridge","#947336"),"target_lgbm":("Local LightGBM","#9b5274")}
fig,axes=plt.subplots(1,2,figsize=(12,4.8),layout="constrained")
for ax,direction in zip(axes,["CHN_to_AUS","AUS_to_CHN"]):
    x=models[(models.direction==direction)&(models.stage=="spatial_temporal_audit")]
    for name,(label,color) in names.items():
        g=x[x.model==name].groupby("window_months").rmse.agg(["median","min","max"])
        ax.plot(g.index,g["median"],marker="o",label=label,color=color)
        ax.fill_between(g.index,g["min"],g["max"],color=color,alpha=.10)
    ax.set(xscale="log",xticks=[1,3,6,12,24],xlabel="Recent months of labels from 10 sentinel cities",
           ylabel="RMSE (ug/m3)",title=direction.replace("_to_"," to "))
    ax.set_xticklabels([1,3,6,12,24]);ax.grid(axis="y",alpha=.15)
axes[0].legend(fontsize=8)
fig.suptitle("Short calibration windows do not establish source-country value\nFull coverage; median and range across three city partitions",fontsize=12)
fig.savefig(OUT/"label_window_baselines.png",dpi=180)
plt.close(fig)

primary=G[(G.seed==5993)&G.stage.eq("spatial_temporal_audit")]
primary[keep].to_csv(OUT/"primary_partition_audit.csv",index=False)
example=Q[(Q.direction=="CHN_to_AUS")&(Q.seed==5993)&(Q.window_months==1)&Q.stage.eq("spatial_temporal_audit")].iloc[0]
no_source=models[(models.direction=="CHN_to_AUS")&(models.seed==5993)&(models.window_months==1)&models.stage.eq("spatial_temporal_audit")&models.model.isin(names)].copy()
no_source.to_csv(OUT/"primary_no_source_ablation.csv",index=False)
coefs=pd.read_csv(HERE/"sequential_results/source_coefficients.csv")
coefs=coefs[(coefs.direction=="CHN_to_AUS")&(coefs.window==1)&(coefs.month_index>=2023*12+1)&coefs.source.eq("source_weighted")]
coefs.groupby("seed").affine_slope.agg(["median","min","max"]).to_csv(OUT/"affine_source_dependence.csv")
summary={"configurations":len(status),"configurations_passing_both":int(status.both_pass.sum()),
         "source_static_manifest":json.loads((HERE/"results/run_manifest.json").read_text()),
         "source_rolling_manifest":json.loads((HERE/"sequential_results/run_manifest.json").read_text()),
         "primary_rolling_one_month_example":example.to_dict(),
         "statistical_status":"Retrospective exploratory; correlated partitions; bootstrap intervals not post-selection guarantees"}
(OUT/"review_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,default=str),encoding="utf-8")
print(counts.to_string())
print(no_source[["model","rmse","r2"]].to_string(index=False))
