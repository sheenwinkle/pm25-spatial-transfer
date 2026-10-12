"""Re-run the discovery search using saved source-label-shuffled predictions."""
from __future__ import annotations

import json
import sys
import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "research" / "thesis_completion_20260912"
DEFAULT_OUT = ROOT / "research" / "manuscript_revision_20261004" / "analysis"
sys.path.insert(0, str(PROJECT))

import run_monitor_models as core


BRANCHES = [
    ("all_networks", "monthly_common"),
    ("all_networks", "monthly_plus_emissions"),
    ("all_networks", "monthly_plus_mixing"),
    ("all_networks", "outer_time_2017_2018"),
    ("hour_auditable", "monthly_common"),
]
PERMUTATIONS = (5993, 5994, 5995)
SEEDS = (5993, 5994, 5995)
WINDOWS = (3, 6, 12)


def run(predictions_root: Path, out_dir: Path, input_label: str) -> pd.DataFrame:
    rows: list[dict] = []
    for cohort, feature_set in BRANCHES:
        for direction in ("AUS_to_CHN", "CHN_to_AUS"):
            folder = predictions_root / cohort / feature_set / direction
            for city_seed in SEEDS:
                for window in WINDOWS:
                    path = folder / f"predictions_{city_seed}_{window}.csv.gz"
                    if not path.exists():
                        raise FileNotFoundError(path)
                    predictions = pd.read_csv(path)
                    discovery = predictions[
                        predictions.development_city
                        & predictions.year.between(2019, 2020)
                    ].copy()
                    for permutation in PERMUTATIONS:
                        candidate = f"shuffled_kernel_{permutation}"
                        controls = [
                            f"shuffled_kernel_{other}"
                            for other in PERMUTATIONS
                            if other != permutation
                        ]
                        core.SOURCE = [candidate]
                        core.NEG = controls
                        core.MODELS = core.LOCAL + [candidate] + controls + ["source_only"]
                        lock, search = core.discover(discovery)
                        rows.append({
                            "cohort": cohort,
                            "feature_set": feature_set,
                            "direction": direction,
                            "city_seed": city_seed,
                            "window_months": window,
                            "null_permutation": permutation,
                            "candidate": candidate,
                            "candidate_rules_evaluated": len(search),
                            "max_point_gain": max(item["gain"] for item in search),
                            "selected_gain": lock["gain"],
                            "selected_coverage": lock["coverage"],
                            "selected_reference": lock["reference"],
                            "selected_rule": json.dumps(lock["rule"], sort_keys=True),
                            "annual_consistency_and_gain_pass": lock["discovery_pass"],
                        })
    result = pd.DataFrame(rows)
    if len(result) != 270:
        raise AssertionError(f"Expected 270 null searches, got {len(result)}")
    out_dir.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_dir / "shuffled_source_discovery_search.csv", index=False)
    summary = {
        "status": "exploratory diagnostic",
        "input": input_label,
        "configuration_count": 90,
        "permutation_count": 3,
        "search_count": int(len(result)),
        "joint_permutation_runs_with_any_discovery_pass": int(
            result.groupby(["cohort", "feature_set", "direction", "city_seed", "window_months"])
            ["annual_consistency_and_gain_pass"].any().sum()
        ),
        "candidate_rules_median": float(result.candidate_rules_evaluated.median()),
        "candidate_rules_range": [
            int(result.candidate_rules_evaluated.min()),
            int(result.candidate_rules_evaluated.max()),
        ],
        "null_search_passes": int(result.annual_consistency_and_gain_pass.sum()),
        "null_search_pass_rate": float(result.annual_consistency_and_gain_pass.mean()),
        "caveats": [
            "Predictions are from the full replay using the corrected NSW December panel.",
            "Only three row-wise source-label permutations are available; these are not independent 270 experiments.",
            "For each null candidate, the other two shuffled kernels and six local learners form the comparator set; the candidate is excluded to avoid self-comparison.",
            "The source-label shuffle breaks source spatial and temporal label structure, so this is a sharp independence null, not a block-preserving null.",
            "Do not interpret the pass rate as an estimate of the fraction of the primary 26/90 discoveries that are false.",
        ],
    }
    (out_dir / "shuffled_source_discovery_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-root", type=Path,
                        default=PROJECT / "results",
                        help="Root containing cohort/feature_set/direction prediction folders")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--input-label", default="saved pre-NSW-correction prediction files")
    args = parser.parse_args()
    data = run(args.predictions_root, args.out_dir, args.input_label)
    print(data.groupby("candidate").agg(
        searches=("candidate", "size"),
        pass_count=("annual_consistency_and_gain_pass", "sum"),
        median_candidates=("candidate_rules_evaluated", "median"),
        median_max_gain=("max_point_gain", "median"),
    ).to_string())
