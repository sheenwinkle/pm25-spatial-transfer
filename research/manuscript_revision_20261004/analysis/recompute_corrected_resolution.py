"""Recompute post-hoc audit-resolution summaries from the corrected replay."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REVISION = ROOT / "research" / "manuscript_revision_20261004"
RESULTS = REVISION / "review_evidence" / "replay" / "corrected_nsw_refit_20261005"
OUT = REVISION / "review_evidence" / "diagnostics" / "corrected_resolution_20261005"
sys.path.insert(0, str(REVISION / "review_evidence" / "diagnostics"))
from common_diag import NEG, Y, bootstrap_deltas  # noqa: E402


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    config_rows: list[dict] = []
    comparator_rows: list[dict] = []
    for manifest_path in sorted(RESULTS.glob("*/*/*/manifest.json")):
        folder = manifest_path.parent
        cohort, feature_set, direction = folder.relative_to(RESULTS).parts
        for seed in (5993, 5994, 5995):
            for window in (3, 6, 12):
                prediction_path = folder / f"predictions_{seed}_{window}.csv.gz"
                prediction = pd.read_csv(prediction_path, float_precision="round_trip")
                lock = json.loads((folder / f"lock_{seed}_{window}.json").read_text(encoding="utf-8"))
                model = lock["model"]
                setting = {
                    "cohort": cohort, "feature_set": feature_set,
                    "direction": direction, "seed": seed,
                    "window_months": window,
                }
                stage_masks = {
                    "temporal_audit": prediction.development_city & prediction.year.ge(2021),
                    "new_city_audit": ~prediction.development_city & prediction.year.ge(2021),
                }
                for stage, stage_mask in stage_masks.items():
                    frame = prediction[stage_mask & prediction.accepted].reset_index(drop=True)
                    if frame.empty:
                        continue
                    references = list(dict.fromkeys([lock["reference"], "target_mean", *NEG]))
                    mdes = []
                    for reference in references:
                        comparator_rmse = float(np.sqrt(np.mean((frame[reference] - frame[Y]) ** 2)))
                        for cluster in ("eFUA_ID", "weather_grid_id"):
                            delta = bootstrap_deltas(frame, model, reference, seed, cluster)
                            if delta is None or len(delta) < 2 or comparator_rmse <= 0:
                                continue
                            se = float(np.std(delta, ddof=1))
                            mde = 100 * 2.8 * se / comparator_rmse
                            mdes.append(mde)
                            comparator_rows.append({
                                **setting, "stage": stage, "candidate": model,
                                "reference": reference, "cluster": cluster,
                                "rows": len(frame), "fuas": int(frame.eFUA_ID.nunique()),
                                "comparator_rmse": comparator_rmse,
                                "bootstrap_se_delta": se, "mde80_pct": mde,
                            })
                    if mdes:
                        config_rows.append({
                            **setting, "stage": stage, "candidate": model,
                            "accepted_rows": len(frame),
                            "accepted_fuas": int(frame.eFUA_ID.nunique()),
                            "conservative_mde80_pct": max(mdes),
                            "mde_reference_cluster_pairs": len(mdes),
                        })

    details = pd.DataFrame(config_rows)
    comparisons = pd.DataFrame(comparator_rows)
    details.to_csv(OUT / "corrected_resolution_by_config.csv", index=False)
    comparisons.to_csv(OUT / "corrected_resolution_by_reference_cluster.csv", index=False)
    summary = details.groupby(["direction", "stage"]).agg(
        configurations=("seed", "size"),
        median_accepted_rows=("accepted_rows", "median"),
        median_accepted_fuas=("accepted_fuas", "median"),
        median_conservative_mde80_pct=("conservative_mde80_pct", "median"),
        p10_mde80_pct=("conservative_mde80_pct", lambda x: x.quantile(0.10)),
        p90_mde80_pct=("conservative_mde80_pct", lambda x: x.quantile(0.90)),
    ).reset_index()
    summary.to_csv(OUT / "corrected_resolution_summary.csv", index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
