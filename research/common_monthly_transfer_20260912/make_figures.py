"""Export the corrected modelling panel and descriptive, unselected audit figures."""
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd

from audit_inputs import HERE
from collect_monthly_power import PROCESSED, VARIABLES, dump, sha

FIG = HERE/"figures"
AUDIT = HERE/"audit"
SETS = ["legacy_annual", "updated_annual", "monthly_common", "monthly_with_aviation"]
LABELS = ["Annual + carry-forward", "Correct-year annual", "Actual monthly", "Monthly + aviation"]
COLORS = ["#63737D", "#B56523", "#007F79", "#B02D55"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 180, "font.family": "DejaVu Sans"})


def save(fig, name):
    fig.savefig(FIG/(name+".png"), facecolor="white")
    fig.savefig(FIG/(name+".pdf"), facecolor="white")
    plt.close(fig)


def main():
    FIG.mkdir(exist_ok=True)
    d, feature_sets, provenance = joblib.load(HERE/"modelling_cache.joblib")
    features = list(dict.fromkeys(c for fs in feature_sets.values() for c in fs))
    columns = ["country_iso", "eFUA_ID", "fua_name", "year", "month", "t", "quarter_id", "weather_grid_id",
               "power_lon", "power_lat", "power_great_circle_distance_km", "pm25_mean_ug_m3"] + features
    panel = PROCESSED/"fua_month_model_panel_2000_2024.csv.gz"
    d[columns].to_csv(panel, index=False, float_format="%.17g")
    assert len(pd.read_csv(panel, usecols=["eFUA_ID"])) == len(d)
    original = {"power_t2m_c": VARIABLES["T2M"], "power_rh2m_percent": VARIABLES["RH2M"],
                "power_ws10m_m_s": VARIABLES["WS10M"], "power_prectotcorr_mm_day": VARIABLES["PRECTOTCORR"]}
    comparisons = []
    for old, new in original.items():
        error = abs(d[old]-d["legacy_annual_"+new])
        comparisons.append(dict(original=old, recreated="legacy_annual_"+new, mean_absolute_difference=error.mean(),
                                max_absolute_difference=error.max(), missing_original=int(d[old].isna().sum())))
        assert np.allclose(d[old], d["legacy_annual_"+new], rtol=1e-6, atol=1e-5)
    pd.DataFrame(comparisons).to_csv(AUDIT/"annual_reconstruction_agreement.csv", index=False)
    d.loc[d.year.ge(2019)].groupby(["country_iso", "year"]).pm25_mean_ug_m3.agg(["size", "mean", "std", "min", "max"]).to_csv(AUDIT/"pm25_yearly_distribution.csv")
    dump(AUDIT/"analysis_ready_export.json", {"file": str(panel), "sha256": sha(panel), "rows": len(d),
        "columns": columns, "pm25_labels_modified": 0, "feature_sets": feature_sets,
        "context": "Existing SatPM FUA means and inherited context, corrected monthly POWER and matched annual ablations. Not an observed station PM2.5 dataset."})

    m = pd.read_csv(AUDIT/"all_model_metrics.csv")
    q = m.loc[m.stage.eq("spatial_temporal_audit") & m.scope.eq("all") & m.window_months.eq(6)]
    models = ["source_partition_kernel", "target_rbf", "target_lgbm", "shuffled_source_kernel"]
    names = ["Source partition", "Local RBF", "Local LightGBM", "Shuffled-source control"]
    marks = ["o", "s", "^", "D"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    for ax, direction, title in zip(axes, ["CHN_to_AUS", "AUS_to_CHN"], ["China to Australia", "Australia to China"]):
        for j, (model, color, marker) in enumerate(zip(models, COLORS, marks)):
            for i, fs in enumerate(SETS):
                values = q.loc[q.direction.eq(direction) & q.model.eq(model) & q.feature_set.eq(fs), "rmse"]
                y = i+(j-1.5)*.16
                mid = values.median()
                ax.errorbar(mid, y, xerr=[[mid-values.min()], [values.max()-mid]], fmt=marker, color=color,
                            capsize=2, markersize=5, linewidth=1.1)
        ax.set_title(title)
        ax.set_xlabel("Held-city RMSE (micrograms per cubic metre)")
        ax.grid(axis="x", alpha=.18)
        ax.set_yticks(range(4), LABELS)
        ax.invert_yaxis()
    fig.suptitle("Matched six-month calibration: full coverage, no gate", y=.98, fontsize=13)
    fig.legend([Line2D([], [], color=c, marker=k, linestyle="") for c,k in zip(COLORS, marks)], names,
               loc="lower center", ncol=4, bbox_to_anchor=(.5, .075), frameon=False)
    fig.text(.5, .025, "Points: median of three overlapping city partitions; bars: range, not confidence intervals. Other windows are retained in the tables.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .16, 1, .94])
    save(fig, "01_matched_weather_ablation")

    g = pd.read_csv(AUDIT/"all_gates.csv")
    q = g.loc[g.stage.eq("spatial_temporal_audit")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.5), sharey=True)
    for ax, direction, title in zip(axes, ["CHN_to_AUS", "AUS_to_CHN"], ["China to Australia", "Australia to China"]):
        for fs, color, label, marker in zip(SETS, COLORS, LABELS, marks):
            part = q.loc[q.direction.eq(direction) & q.feature_set.eq(fs)]
            ax.scatter(part.coverage*100, 1-part.gain_vs_all_comparators, c=color, marker=marker, s=37, alpha=.75, label=label)
        ax.axhline(1, color="#444444", linewidth=1)
        ax.axhline(.95, color="#007F79", linestyle="--", linewidth=1)
        ax.axvline(25, color="#777777", linestyle=":", linewidth=1)
        ax.set_yscale("log"); ax.set_ylim(.85, 5); ax.set_xlim(10, 103)
        ax.set_yticks([1, 1.5, 2, 3, 5], ["1.00", "1.50", "2.00", "3.00", "5.00"])
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title); ax.set_xlabel("Fraction of all held-city months accepted (%)")
        ax.grid(alpha=.15)
    axes[0].set_ylabel("Frozen model RMSE / strongest comparator RMSE (log scale)")
    fig.suptitle("All 96 frozen configurations: coverage and transfer gain", y=.98, fontsize=13)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(.5,.075), frameon=False)
    fig.text(.5, .025, "Lower is better. Dashed line: 5% improvement; dotted line: 25% minimum coverage. No configuration crosses the 5% improvement line.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0, .16, 1, .94])
    save(fig, "02_frozen_gate_frontier")

    case = g.loc[g.feature_set.eq("monthly_with_aviation") & g.direction.eq("AUS_to_CHN") & g.seed.eq(5995) & g.window_months.eq(6)]
    order = ["discovery", "confirmation", "spatial_temporal_audit", "seen_city_temporal_audit", "new_weather_grid_audit"]
    labels = ["2019-20 discovery", "2021-22 same network", "2023-24 other cities", "2023-24 same network", "2023-24 other weather cells"]
    case = case.set_index("stage").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5), gridspec_kw={"width_ratios": [1.3, 1]})
    delta = case.target_lgbm_rmse-case.rmse
    axes[0].errorbar(delta, range(5), xerr=[delta-case.fua_frozen_delta_low, case.fua_frozen_delta_high-delta],
                     fmt="o", color="#007F79", capsize=4)
    axes[0].axvline(0, color="#777777", linestyle="--")
    axes[0].set_yticks(range(5), labels); axes[0].invert_yaxis()
    axes[0].set_xlabel("Local LightGBM RMSE minus transfer RMSE\n95% exploratory FUA x quarter interval")
    axes[1].barh(range(5), case.coverage*100, color="#63737D", height=.5)
    axes[1].set_yticks(range(5), [""]*5); axes[1].invert_yaxis(); axes[1].set_xlim(0, 65)
    for i, value in enumerate(case.coverage*100): axes[1].text(value+1, i, f"{value:.1f}%", va="center")
    axes[1].set_xlabel("Accepted rows (%)")
    fig.suptitle("Candidate temperature interval: 12.63-25.13 C (Australia to China)", fontsize=13)
    fig.text(.5, .035, "Case highlighted after inspecting outcomes, not a newly validated optimum. Every interval crosses zero; do not present it as an established transfer region.", ha="center", fontsize=9)
    fig.tight_layout(rect=[0,.09,1,.94])
    save(fig, "03_candidate_interval_audit")
    print("FIGURES_AND_PANEL_COMPLETE", flush=True)


if __name__ == "__main__": main()
