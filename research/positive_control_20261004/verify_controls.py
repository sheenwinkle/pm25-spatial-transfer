"""Independent output checks for the procedural and synthetic controls."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_positive_control as pc

HERE = Path(__file__).resolve().parent


def close(left, right, tolerance=1e-10):
    return bool(np.isclose(float(left), float(right), rtol=0, atol=tolerance))


def verify_family(name, result_root, audit_root, script, protocol):
    gates = pd.read_csv(audit_root / "all_gates.csv")
    checked = 0
    for seed in pc.SEEDS:
        for window in pc.WINDOWS:
            folder = result_root / str(seed) / str(window)
            predictions = pd.read_csv(folder / "predictions.csv.gz")
            lock = json.loads((folder / "lock.json").read_text(encoding="utf-8"))
            accepted = pc.core.maskrule(predictions, lock["rule"])
            assert np.array_equal(accepted, predictions.accepted.to_numpy(dtype=bool))
            assert np.allclose(
                predictions.selected_prediction,
                predictions[lock["model"]],
                rtol=0,
                atol=1e-12,
            )
            stages = {
                "discovery": predictions.development_city & predictions.year.le(2020),
                "temporal_audit": predictions.development_city & predictions.year.ge(2021),
                "new_city_audit": ~predictions.development_city & predictions.year.ge(2021),
                "new_weather_grid_audit": (
                    ~predictions.development_city
                    & predictions.new_weather_grid
                    & predictions.year.ge(2021)
                ),
            }
            rows = gates[(gates.seed == seed) & (gates.window_months == window)]
            assert set(rows.stage) == set(stages)
            for row in rows.itertuples(index=False):
                full = predictions[stages[row.stage]]
                selected = full[full.accepted]
                metrics = pc.core.metrics(selected, selected[lock["model"]])
                relative_gain, best, _ = pc.core.gain(
                    full, lock["model"], full.accepted.to_numpy(dtype=bool)
                )
                assert len(selected) == row.n
                assert selected.eFUA_ID.nunique() == row.fuas
                assert selected.t.nunique() == row.months
                assert close(metrics["rmse"], row.rmse)
                assert close(relative_gain, row.relative_gain)
                assert best == row.best_comparator
                assert close(len(selected) / len(full), row.coverage)
                checked += 1
            fit_audit = pd.read_csv(folder / "fit_audit.csv")
            assert (fit_audit.latest_fit_t < fit_audit.prediction_t).all()
            complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
            assert complete["protocol_sha256"] == pc.digest(protocol)
            assert complete["script_sha256"] == pc.digest(script)
            assert complete["predictions_sha256"] == pc.digest(folder / "predictions.csv.gz")
    completion = json.loads((audit_root / "completion.json").read_text(encoding="utf-8"))
    assert completion["protocol_sha256"] == pc.digest(protocol)
    assert completion["script_sha256"] == pc.digest(script)
    return {"family": name, "gate_stage_records_checked": checked, **completion}


def main():
    reports = [
        verify_family(
            "within_china_procedural",
            HERE / "results",
            HERE / "audit",
            HERE / "run_positive_control.py",
            HERE / "PROTOCOL.md",
        ),
        verify_family(
            "synthetic_known_effect",
            HERE / "synthetic_results",
            HERE / "synthetic_audit",
            HERE / "run_synthetic_recovery.py",
            HERE / "SYNTHETIC_PROTOCOL.md",
        ),
    ]
    synthetic_completion = reports[1]
    label_path = HERE / "synthetic_data/synthetic_target_labels.csv.gz"
    generation = json.loads(
        (HERE / "synthetic_audit/generation_receipt.json").read_text(encoding="utf-8")
    )
    assert generation["label_sha256"] == pc.digest(label_path)
    assert synthetic_completion["label_sha256"] == pc.digest(label_path)
    output = {"status": "passed", "families": reports}
    pc.dump(HERE / "audit/verification.json", output)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
