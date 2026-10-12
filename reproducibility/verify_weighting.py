"""Verify retained weighting predictions without requiring a manuscript file."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
AUDIT = REPO / "research/manuscript_revision_20261004/audit"
SW = "NNW[LDF] (sample-weight)"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(old_root, new_root, out):
    for folder in (old_root, new_root):
        if not folder.is_dir():
            raise FileNotFoundError(
                f"Prediction directory not found: {folder}. See reproducibility/README.md; "
                "row-level prediction archives are not distributed in this repository."
            )
    files = sorted(new_root.glob("*/*/seed*_predictions.csv.gz"))
    require(len(files) == 200, f"Expected 200 prediction files, found {len(files)}")
    metric_count, recomputed_count, worst = 0, 0, 0.0
    for file in files:
        relative = file.relative_to(new_root)
        old = pd.read_csv(old_root / relative, float_precision="round_trip")
        new = pd.read_csv(file, float_precision="round_trip")
        require(SW in new.columns, f"Missing loss-weighted model: {relative}")
        require(list(new.drop(columns=SW).columns) == list(old.columns),
                f"Original columns changed: {relative}")
        pd.testing.assert_frame_equal(old, new[old.columns], check_exact=True)
        name = file.name.replace("_predictions.csv.gz", "_metrics.json")
        old_meta = json.loads((old_root / relative.parent / name).read_text(encoding="utf-8"))
        new_meta = json.loads((file.parent / name).read_text(encoding="utf-8"))
        require(set(new_meta["scores"]) == set(old_meta["scores"]) | {SW},
                f"Unexpected model scores: {relative}")
        for model, score in old_meta["scores"].items():
            require(score == new_meta["scores"][model], f"Original scores changed: {relative}, {model}")
            metric_count += 1
        y = new["observed_pm25"].to_numpy()
        for model, score in new_meta["scores"].items():
            value = float(np.sqrt(np.mean((new[model].to_numpy() - y) ** 2)))
            error = abs(value - score["rmse"])
            require(np.isfinite(error) and error < 1e-12,
                    f"RMSE mismatch: {relative}, {model}, error={error}")
            worst = max(worst, error)
            recomputed_count += 1
    require(metric_count == 2200 and recomputed_count == 2400,
            f"Unexpected metric counts: {metric_count}, {recomputed_count}")

    # Reuse the published bootstrap implementation and compare its saved precision.
    out.mkdir(parents=True, exist_ok=True)
    summary = AUDIT / "weighting_check_20261012/window_weighting_summary.py"
    with (out / "summary.log").open("w", encoding="utf-8") as log:
        subprocess.run([sys.executable, str(summary), str(new_root), str(out / "recomputed")],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    tables = {}
    for name in ("gains", "model_means"):
        reference = pd.read_csv(AUDIT / f"weighting_check_20261012/weighting_{name}.csv")
        tables[name] = pd.read_csv(out / f"recomputed_{name}.csv")
        pd.testing.assert_frame_equal(reference, tables[name], check_exact=True)
    gains = tables["gains"]
    strong = gains[(gains.comparison == "hyb_vs_stronger") & (gains.ldf_seed_offset == 0)]
    recent = strong[strong.window.str.startswith("2020")]
    require(len(recent) == 6 and int((recent.lo > 0).sum()) == 5,
            "The six 2020 comparisons do not reproduce the expected five positive intervals")
    report = {
        "prediction_files_compared_exactly": len(files),
        "unchanged_original_model_scores": metric_count,
        "rmses_recomputed": recomputed_count,
        "maximum_absolute_rmse_error": worst,
        "bootstrap_comparisons_reproduced_at_saved_precision": len(gains),
        "model_mean_and_rank_rows_reproduced": len(tables["model_means"]),
        "hybrid_vs_stronger_2020_positive_intervals": "5/6",
        "scope": "Saved-result verification, not model training; bootstrap reuses the published summary implementation.",
    }
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, required=True, help="Original window_results directory")
    parser.add_argument("--new", type=Path, required=True, help="Loss-weighted window_results_sw directory")
    parser.add_argument("--out", type=Path, default=REPO / "reproducibility/output",
                        help="Directory for regenerated summaries and verification.json")
    args = parser.parse_args()
    report = verify(args.old.resolve(), args.new.resolve(), args.out.resolve())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
