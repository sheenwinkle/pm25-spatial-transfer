"""Produce a paired, ungated model comparison for the post-hoc China control."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
INPUT = ROOT / "research" / "positive_control_20261004" / "audit"
OUTPUT = ROOT / "research" / "manuscript_revision_20261004" / "analysis"
CONTROL = "nonYRD_to_YRD"
LOCAL = ["target_mean", "target_ridge", "target_rbf", "target_rf", "target_extra", "target_lgbm"]
SOURCE = ["source_affine", "source_residual", "source_kernel", "weighted_source_kernel", "source_only"]
STAGES = ["temporal_audit", "new_city_audit"]


def main() -> None:
    metrics = pd.read_csv(INPUT / "all_metrics.csv.gz")
    subset = metrics[
        metrics.control.eq(CONTROL)
        & metrics.window_months.eq(6)
        & metrics.selection.eq("full")
        & metrics.stage.isin(STAGES)
    ].copy()
    wide = subset.pivot(index=["stage", "seed"], columns="model", values="rmse")
    wide_r2 = subset.pivot(index=["stage", "seed"], columns="model", values="r2")
    best_local = wide[LOCAL].min(axis=1)
    best_local_name = wide[LOCAL].idxmin(axis=1)
    rows = []
    for (stage, seed), values in wide.iterrows():
        for model in SOURCE:
            rows.append({
                "control": CONTROL,
                "post_hoc": True,
                "window_months": 6,
                "stage": stage,
                "seed": int(seed),
                "model": model,
                "rmse": float(values[model]),
                "r2": float(wide_r2.loc[(stage, seed), model]),
                "best_local_model": str(best_local_name.loc[(stage, seed)]),
                "best_local_rmse": float(best_local.loc[(stage, seed)]),
                "paired_relative_rmse_gain": float(1 - values[model] / best_local.loc[(stage, seed)]),
            })
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT / "within_china_ungated_6month_by_seed.csv", index=False)
    summary = (result.groupby(["stage", "model"], as_index=False)
               .agg(rmse_median=("rmse", "median"), r2_median=("r2", "median"),
                    paired_gain_median=("paired_relative_rmse_gain", "median"),
                    paired_gain_min=("paired_relative_rmse_gain", "min"),
                    paired_gain_max=("paired_relative_rmse_gain", "max")))
    summary.to_csv(OUTPUT / "within_china_ungated_6month_summary.csv", index=False)
    gates = pd.read_csv(INPUT / "all_gates.csv")
    selected = gates[gates.control.eq(CONTROL) & gates.stage.eq("discovery")].copy()
    chosen = selected.selected_model.value_counts().to_dict()
    six_month = selected[selected.window_months.eq(6)][
        ["seed", "selected_model", "discovery_pass", "relative_gain", "best_comparator"]
    ].sort_values("seed")
    out = {
        "status": "post-hoc descriptive positive-control reanalysis",
        "control": CONTROL,
        "window_months": 6,
        "rows": len(result),
        "per_seed_table": "within_china_ungated_6month_by_seed.csv",
        "summary_table": "within_china_ungated_6month_summary.csv",
        "discovery_selected_model_counts_all_9_configs": chosen,
        "six_month_discovery_selections": six_month.to_dict(orient="records"),
        "interpretation": "Ungated estimator-level RMSE gains do not equal a validated positive result for the full adaptive gate and joint-audit procedure.",
        "new_south_wales_correction_affects_this_control": False,
    }
    (OUTPUT / "within_china_control_reanalysis.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    print("\nSelected discovery models:", chosen)
    print(six_month.to_string(index=False))


if __name__ == "__main__":
    OUTPUT.mkdir(parents=True, exist_ok=True)
    main()
