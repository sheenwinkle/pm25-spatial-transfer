"""Vectorised station-to-FUA aggregation for the monthly GSOD panel."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STATION_PATH = ROOT / "data/interim/aviation/gsod_station_month_2000_2025.csv.gz"
CANDIDATES_PATH = ROOT / "data/interim/aviation/isd_station_candidates_2000_2025.csv"
OUTPUT_PATH = ROOT / "data/interim/aviation/gsod_fua_month_2000_2025.csv.gz"
METADATA_PATH = ROOT / "data/metadata/gsod_monthly_surface_panel_2000_2025.json"
MANIFEST_PATH = ROOT / "data/metadata/gsod_download_manifest_2000_2025.csv"

KEYS = ["eFUA_ID", "eFUA_name", "Cntry_ISO", "year", "month"]
EXCLUDED_VALUE_COLUMNS = {
    "station_id",
    "year",
    "month",
    "station_month_eligible",
    "dry_visibility_month_eligible",
    "daily_record_count",
}


def grouped_nunique(frame: pd.DataFrame, column: str, output: str) -> pd.DataFrame:
    return frame.groupby(KEYS, observed=True)[column].nunique().rename(output).reset_index()


def build_panel(station: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    station["station_id"] = station["station_id"].astype(str).str.zfill(11)
    station_map = candidates[
        ["station_id", "eFUA_ID", "eFUA_name", "Cntry_ISO", "is_aviation_icao", "distance_to_fua_km"]
    ].copy()
    station_map["station_id"] = station_map["station_id"].astype(str).str.zfill(11)
    merged = station.merge(station_map, on="station_id", how="inner", validate="many_to_one")
    merged["is_aviation_icao"] = merged["is_aviation_icao"].fillna(False).astype(bool)

    all_groups = merged.groupby(KEYS, observed=True, sort=False)
    base = all_groups.agg(
        surface_station_count=("station_id", "nunique"),
        nearest_surface_station_km=("distance_to_fua_km", "min"),
        median_surface_station_km=("distance_to_fua_km", "median"),
    ).reset_index()

    aviation = merged.loc[merged["is_aviation_icao"]]
    base = base.merge(
        grouped_nunique(aviation, "station_id", "aviation_icao_station_count"),
        on=KEYS,
        how="left",
    )

    eligible = merged.loc[merged["station_month_eligible"]].copy()
    eligible_groups = eligible.groupby(KEYS, observed=True, sort=False)
    eligible_base = eligible_groups.agg(
        surface_eligible_station_count=("station_id", "nunique"),
        surface_valid_station_days=("daily_record_count", "sum"),
    ).reset_index()
    eligible_base["surface_source_available"] = True
    base = base.merge(eligible_base, on=KEYS, how="left")

    eligible_aviation = eligible.loc[eligible["is_aviation_icao"]]
    eligible_aviation_count = grouped_nunique(
        eligible_aviation, "station_id", "aviation_icao_eligible_station_count"
    )
    eligible_aviation_count["aviation_icao_source_available"] = True
    base = base.merge(eligible_aviation_count, on=KEYS, how="left")

    value_columns = [column for column in station.columns if column not in EXCLUDED_VALUE_COLUMNS]
    values = eligible[value_columns].apply(pd.to_numeric, errors="coerce")
    weights = eligible["daily_record_count"].astype(float)
    valid = values.notna()
    numerators = values.mul(weights, axis=0).where(valid, 0.0)
    denominators = valid.mul(weights, axis=0)
    numerators[KEYS] = eligible[KEYS].to_numpy()
    denominators[KEYS] = eligible[KEYS].to_numpy()
    numerator_sums = numerators.groupby(KEYS, observed=True, sort=False)[value_columns].sum()
    denominator_sums = denominators.groupby(KEYS, observed=True, sort=False)[value_columns].sum()
    weighted = numerator_sums.div(denominator_sums.replace(0, np.nan))
    weighted.columns = [f"surface_{column}" for column in weighted.columns]
    base = base.merge(weighted.reset_index(), on=KEYS, how="left")

    count_columns = [
        "surface_station_count",
        "aviation_icao_station_count",
        "surface_eligible_station_count",
        "aviation_icao_eligible_station_count",
        "surface_valid_station_days",
    ]
    base[count_columns] = base[count_columns].fillna(0).astype(int)
    for column in ["surface_source_available", "aviation_icao_source_available"]:
        base[column] = base[column].fillna(False).astype(bool)
    return base.sort_values(["Cntry_ISO", "eFUA_ID", "year", "month"]).reset_index(drop=True)


def main() -> int:
    print("reading completed station-month panel", flush=True)
    station = pd.read_csv(STATION_PATH, dtype={"station_id": str})
    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"station_id": str})
    print(f"aggregating {len(station):,} station-month rows", flush=True)
    panel = build_panel(station, candidates)
    panel.to_csv(OUTPUT_PATH, index=False, compression="gzip", encoding="utf-8")

    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    manifest = pd.read_csv(MANIFEST_PATH) if MANIFEST_PATH.exists() else pd.DataFrame()
    metadata.update(
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "NOAA NCEI Global Summary of the Day, derived from Integrated Surface Hourly",
            "source_url": "https://www.ncei.noaa.gov/data/global-summary-of-the-day/",
            "study_period": [2000, 2025],
            "candidate_stations": int(candidates["station_id"].nunique()),
            "expected_station_year_files": int(len(manifest)) if not manifest.empty else None,
            "available_station_year_files": int((manifest["status"] == "existing").sum())
            if not manifest.empty
            else None,
            "download_status_counts": manifest["status"].value_counts().to_dict()
            if not manifest.empty
            else {},
            "station_month_rows": int(len(station)),
            "fua_month_rows_with_surface_source": int(len(panel)),
            "fua_count_with_surface_source": int(panel["eFUA_ID"].nunique()),
            "minimum_valid_days": 15,
            "dry_visibility_rule": "visibility present; RH < 80%; no fog, rain/drizzle, or snow/ice flag",
            "outputs": [
                str(STATION_PATH.relative_to(ROOT)),
                str(OUTPUT_PATH.relative_to(ROOT)),
                str(MANIFEST_PATH.relative_to(ROOT)),
            ],
            "fua_aggregation": "vectorised daily-record-count-weighted means across eligible stations",
        }
    )
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
