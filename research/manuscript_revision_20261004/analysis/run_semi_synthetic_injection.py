"""Run frozen source-response blend diagnostics on corrected YRD data."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
POSITIVE_CONTROL = ROOT / "research" / "positive_control_20261004"
sys.path.insert(0, str(POSITIVE_CONTROL))

import run_synthetic_recovery as runner  # noqa: E402
import run_positive_control as pc  # noqa: E402

PROTOCOL = ROOT / "research" / "manuscript_revision_20261004" / "review_evidence" / "diagnostics" / "semi_synthetic_injection_protocol.md"
STRENGTHS = (0.05, 0.10, 0.20, 0.40)


def run_level(strength: float, root: Path) -> None:
    label = f"blend_{int(round(strength * 100)):02d}pct"
    folder = root / label
    if (folder / "audit" / "completion.json").exists():
        raise FileExistsError(f"Completed result already exists: {folder}")
    runner.DATA = folder / "data"
    runner.AUDIT = folder / "audit"
    runner.RESULTS = folder / "results"
    for path in (runner.DATA, runner.AUDIT, runner.RESULTS):
        path.mkdir(parents=True, exist_ok=True)

    def generate_target(source, target, features, all_path, membership_path):
        source_model = pc.core.tree(True).fit(source[features], source[pc.Y])
        source_signal = pc.core.pos(source_model.predict(target[features]))
        observed = target[pc.Y].to_numpy(dtype=float)
        semi_synthetic = pc.core.pos(
            (1.0 - strength) * observed + strength * source_signal
        )
        labels = target[["eFUA_ID", "year", "month", "t"]].copy()
        labels["observed_pm25"] = observed
        labels["source_signal"] = source_signal
        labels["semi_synthetic_pm25"] = semi_synthetic
        labels["injection_proportion"] = strength
        label_path = runner.DATA / "synthetic_target_labels.csv.gz"
        labels.to_csv(label_path, index=False)
        pc.dump(runner.AUDIT / "generation_receipt.json", {
            "protocol_sha256": pc.digest(PROTOCOL),
            "script_sha256": pc.digest(__file__),
            "all_network_panel_sha256": pc.digest(all_path),
            "membership_panel_sha256": pc.digest(membership_path),
            "label_sha256": pc.digest(label_path),
            "injection_proportion": strength,
            "source_rows": int(len(source)),
            "target_rows": int(len(target)),
            "target_fuas": int(target.eFUA_ID.nunique()),
            "observed_target_mean": float(np.mean(observed)),
            "source_signal_mean": float(np.mean(source_signal)),
            "semi_synthetic_mean": float(np.mean(semi_synthetic)),
            "source_response_blend_is_not_an_rmse_gain": True,
        })
        target[pc.Y] = semi_synthetic
        return target

    runner.generate_target = generate_target
    runner.run()
    print("INJECTION_COMPLETE", label, flush=True)


def summarize(root: Path) -> pd.DataFrame:
    rows: list[dict] = []
    for strength in STRENGTHS:
        label = f"blend_{int(round(strength * 100)):02d}pct"
        audit = root / label / "audit"
        gates = pd.read_csv(audit / "all_gates.csv")
        metrics = pd.read_csv(audit / "all_metrics.csv.gz")
        wide = gates.pivot(index=["seed", "window_months"], columns="stage", values="pass")
        joint = wide["temporal_audit"] & wide["new_city_audit"]
        temporal = metrics[
            metrics.stage.eq("temporal_audit") & metrics.selection.eq("full")
        ]
        new_city = metrics[
            metrics.stage.eq("new_city_audit") & metrics.selection.eq("full")
        ]
        for stage, table in (("temporal_audit", temporal), ("new_city_audit", new_city)):
            for (seed, window), group in table.groupby(["seed", "window_months"]):
                by_model = group.set_index("model")
                local_rmse = by_model.loc[list(pc.core.LOCAL), "rmse"].min()
                source_rmse = by_model.loc["source_affine", "rmse"]
                rows.append({
                    "injection_proportion": strength,
                    "stage": stage,
                    "seed": int(seed),
                    "window_months": int(window),
                    "source_affine_rmse": float(source_rmse),
                    "best_local_rmse": float(local_rmse),
                    "source_affine_paired_gain_pct": float(100 * (1 - source_rmse / local_rmse)),
                    "selected_gate_joint_pass": bool(joint.loc[(seed, window)]),
                })
    summary = pd.DataFrame(rows)
    summary.to_csv(root / "semi_synthetic_injection_by_config.csv", index=False)
    aggregate = summary.groupby(["injection_proportion", "stage"]).agg(
        configurations=("seed", "size"),
        median_affine_gain_pct=("source_affine_paired_gain_pct", "median"),
        min_affine_gain_pct=("source_affine_paired_gain_pct", "min"),
        max_affine_gain_pct=("source_affine_paired_gain_pct", "max"),
        joint_passes=("selected_gate_joint_pass", "sum"),
    ).reset_index()
    aggregate.to_csv(root / "semi_synthetic_injection_summary.csv", index=False)
    receipt = {
        "status": "complete",
        "protocol_sha256": pc.digest(PROTOCOL),
        "script_sha256": pc.digest(__file__),
        "injection_proportions": list(STRENGTHS),
        "interpretation": "Retrospective semi-synthetic stress diagnostic; not prospective power or inferential validation.",
    }
    (root / "completion.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    return aggregate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "research" / "manuscript_revision_20261004" / "review_evidence" / "replay" / "semi_synthetic_injection_20261005",
    )
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        for strength in STRENGTHS:
            run_level(strength, args.out_dir)
    print(summarize(args.out_dir).to_string(index=False))


if __name__ == "__main__":
    main()
