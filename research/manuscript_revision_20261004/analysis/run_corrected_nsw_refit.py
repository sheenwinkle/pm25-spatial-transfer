"""Refit the unchanged transfer protocol on the corrected NSW target panel."""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REVISION = ROOT / "research" / "manuscript_revision_20261004"
PROJECT = ROOT / "research" / "thesis_completion_20260912"
CORRECTED_PANEL = ROOT / "research" / "boundary_regression_20261001" / "data" / "all_networks_panel.csv.gz"
OUTROOT = REVISION / "review_evidence" / "replay" / "corrected_nsw_refit_20261005"
sys.path.insert(0, str(PROJECT))

import run_monitor_models as primary
import run_supplement_models as supplement

BRANCHES = [
    ("all_networks", "monthly_common"),
    ("all_networks", "monthly_plus_emissions"),
    ("all_networks", "monthly_plus_mixing"),
    ("all_networks", "outer_time_2017_2018"),
    ("hour_auditable", "monthly_common"),
]
DIRECTIONS = ("AUS_to_CHN", "CHN_to_AUS")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_and_validate_panels() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    features = json.loads((PROJECT / "audit" / "build_summary.json").read_text(encoding="utf-8"))["features"]
    corrected = pd.read_csv(CORRECTED_PANEL)
    original = pd.read_csv(PROJECT / "data" / "all_networks_model_panel.csv.gz")
    keys = ["country_iso", "eFUA_ID", "year", "month"]
    if len(corrected) != 34639 or corrected.duplicated(keys).any():
        raise AssertionError("Corrected panel row count or country-city-month key is invalid")
    if len(original) != len(corrected):
        raise AssertionError("Original and corrected panels differ in row count")
    joined = original[keys + ["observed_pm25"]].merge(
        corrected[keys + ["observed_pm25"]], on=keys, suffixes=("_old", "_new"), validate="one_to_one"
    )
    changed = ~np.isclose(joined.observed_pm25_old, joined.observed_pm25_new, rtol=0, atol=1e-12)
    if int(changed.sum()) != 32:
        raise AssertionError(f"Expected 32 corrected target values, found {int(changed.sum())}")
    for feature in features:
        if feature not in corrected:
            raise AssertionError(f"Missing common feature: {feature}")
    required = ["temperature_c", "humidity_pct", "wind_mps", "monthly_power_precip_mm_day",
                "distance_to_coast_km", "log_fire_count_100km", "log_population_density", "weather_grid_id"]
    if any(column not in corrected for column in required):
        raise AssertionError("Corrected panel is missing gate or spatial-support fields")
    hour = pd.read_csv(PROJECT / "data" / "hour_auditable_model_panel.csv.gz")
    return corrected, hour, features


def fit_spec(cohort: str, feature_set: str, direction: str, corrected: pd.DataFrame,
             hour: pd.DataFrame, common_features: list[str]) -> None:
    folder = OUTROOT / cohort / feature_set / direction
    if (folder / "manifest.json").exists():
        print("SKIP existing", cohort, feature_set, direction, flush=True)
        return
    data = corrected.copy() if cohort == "all_networks" else hour.copy()
    if feature_set in ("monthly_common", "monthly_plus_emissions"):
        features = list(common_features)
        if feature_set == "monthly_plus_emissions":
            features += [column for column in data if column.startswith("log_edgar_")]
        primary.run(data, features, cohort, feature_set, direction)
        return
    if cohort != "all_networks":
        raise ValueError(f"Unsupported cohort/feature branch: {cohort}/{feature_set}")
    features = list(common_features)
    if feature_set == "outer_time_2017_2018":
        supplement.SOURCE_END, supplement.DISC_FIRST = 2016, 2017
        supplement.AUDIT_FIRST, supplement.AUDIT_END = 2019, 2020
    elif feature_set == "monthly_plus_mixing":
        supplement.SOURCE_END, supplement.DISC_FIRST = 2018, 2019
        supplement.AUDIT_FIRST, supplement.AUDIT_END = 2021, 2022
        cols = ["era5_pbl_mean_m", "era5_pbl_low_fraction", "era5_surface_ventilation_mean_m2_s"]
        log_cols = ["log_era5_pbl_mean_m", "log_era5_surface_ventilation_mean_m2_s"]
        data = data.drop(columns=[c for c in cols + log_cols if c in data])
        mix = pd.read_csv(PROJECT / "data" / "common_era5_month_2015_2022.csv.gz")
        data = data.merge(mix[["eFUA_ID", "year", "month"] + cols],
                          on=["eFUA_ID", "year", "month"], validate="one_to_one")
        if not data[cols].notna().all().all():
            raise AssertionError("ERA5 mixing-height features have missing values")
        for column in (cols[0], cols[2]):
            data["log_" + column] = np.log1p(data[column])
            features.append("log_" + column)
        features.append(cols[1])
    else:
        raise ValueError(f"Unsupported feature set: {feature_set}")
    supplement.run(data, features, cohort, feature_set, direction)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", action="append", default=[], help="cohort/feature_set/direction; repeatable")
    args = parser.parse_args()
    OUTROOT.mkdir(parents=True, exist_ok=True)
    corrected, hour, features = load_and_validate_panels()
    primary.RESULTS = OUTROOT
    supplement.RESULTS = OUTROOT
    specs = args.spec or [f"{cohort}/{feature_set}/{direction}"
                          for cohort, feature_set in BRANCHES for direction in DIRECTIONS]
    started = time.time()
    log_path = OUTROOT / "run_log.csv"
    for spec in specs:
        cohort, feature_set, direction = spec.split("/")
        if (cohort, feature_set) not in BRANCHES or direction not in DIRECTIONS:
            raise ValueError(f"Unrecognised experiment spec: {spec}")
        before = time.time()
        fit_spec(cohort, feature_set, direction, corrected, hour, features)
        elapsed = time.time() - before
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(f"{spec},{elapsed:.1f},{time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n")
        print("DONE", spec, f"{elapsed:.1f}s", flush=True)
    record = {
        "status": "complete_for_requested_specs",
        "input_corrected_panel": str(CORRECTED_PANEL),
        "input_sha256": sha256(CORRECTED_PANEL),
        "corrected_panel_rows": int(len(corrected)),
        "corrected_target_values": 32,
        "common_features": features,
        "requested_specs": specs,
        "elapsed_seconds": round(time.time() - started, 1),
        "python": sys.version,
    }
    (OUTROOT / "run_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
