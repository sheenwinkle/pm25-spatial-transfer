from __future__ import annotations

import json
import math
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


START_YEAR = 2000
END_YEAR = 2025
VARIABLES = ["ghgt", "temp", "uwnd", "vapr", "vwnd"]
HOURS = ["00", "12"]
LEVELS = [9999, 1000, 925, 850, 700, 500]
BASE_URL = "https://www.ncei.noaa.gov/pub/data/igra/monthly/monthly-por"

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/interim/aviation/igra_station_candidates_2000_2025.csv"
RAW_DIR = ROOT / "data/raw/meteorology/noaa_igra_monthly_por"
INTERIM_DIR = ROOT / "data/interim/aviation"
META_DIR = ROOT / "data/metadata"
STATION_MONTH_PATH = INTERIM_DIR / "igra_station_month_2000_2025.csv.gz"
FUA_MONTH_PATH = INTERIM_DIR / "igra_fua_month_2000_2025.csv.gz"
METADATA_PATH = META_DIR / "igra_monthly_upper_air_panel_2000_2025.json"


def download_one(variable: str, hour: str, retries: int = 4) -> dict[str, object]:
    name = f"{variable}_{hour}z-mly.txt.zip"
    path = RAW_DIR / name
    url = f"{BASE_URL}/{name}"
    if path.exists() and path.stat().st_size > 1_000_000:
        return {"file": name, "status": "existing", "size_bytes": path.stat().st_size, "url": url}
    partial = path.with_suffix(path.suffix + ".part")
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if partial.stat().st_size < 1_000_000:
                raise ValueError("IGRA archive was unexpectedly small")
            partial.replace(path)
            return {"file": name, "status": "downloaded", "size_bytes": path.stat().st_size, "url": url}
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if partial.exists():
                partial.unlink()
            time.sleep(min(2**attempt, 10))
    return {"file": name, "status": "failed", "size_bytes": 0, "url": url, "error": last_error}


def download_archives() -> list[dict[str, object]]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(download_one, variable, hour): (variable, hour)
            for variable in VARIABLES
            for hour in HOURS
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"IGRA archive {record['file']}: {record['status']}", flush=True)
    return records


def parse_archive(variable: str, hour: str, station_ids: set[str]) -> pd.DataFrame:
    path = RAW_DIR / f"{variable}_{hour}z-mly.txt.zip"
    rows: list[tuple[str, int, int, int, float, int, str, str]] = []
    with zipfile.ZipFile(path) as archive:
        member = archive.namelist()[0]
        with archive.open(member) as stream:
            for raw_line in stream:
                station_id = raw_line[0:11].decode("ascii")
                if station_id not in station_ids:
                    continue
                try:
                    year = int(raw_line[12:16])
                    month = int(raw_line[17:19])
                    level = int(raw_line[20:24])
                    value = int(raw_line[25:31])
                    count = int(raw_line[32:34])
                except ValueError:
                    continue
                if START_YEAR <= year <= END_YEAR and level in LEVELS and count >= 10:
                    scale = 0.1 if variable in {"temp", "uwnd", "vwnd"} else 1.0
                    rows.append((station_id, year, month, level, value * scale, count, variable, hour))
    print(f"parsed {path.name}: {len(rows):,} selected records", flush=True)
    return pd.DataFrame(
        rows,
        columns=["igra_id", "year", "month", "level_hpa", "value", "observation_count", "variable", "hour"],
    )


def build_station_month(candidates: pd.DataFrame) -> pd.DataFrame:
    station_ids = set(candidates["igra_id"].astype(str))
    frames = [parse_archive(variable, hour, station_ids) for variable in VARIABLES for hour in HOURS]
    long = pd.concat(frames, ignore_index=True)
    long["weighted_value"] = long["value"] * long["observation_count"]
    combined = long.groupby(
        ["igra_id", "year", "month", "level_hpa", "variable"], as_index=False
    ).agg(weighted_value=("weighted_value", "sum"), observation_count=("observation_count", "sum"))
    combined["value"] = combined["weighted_value"] / combined["observation_count"]
    combined["feature"] = combined["variable"] + "_" + combined["level_hpa"].astype(str)

    values = combined.pivot(index=["igra_id", "year", "month"], columns="feature", values="value")
    counts = combined.pivot(
        index=["igra_id", "year", "month"], columns="feature", values="observation_count"
    ).add_prefix("n_")
    panel = pd.concat([values, counts], axis=1).reset_index()
    panel.columns.name = None

    for level in LEVELS:
        u_column = f"uwnd_{level}"
        v_column = f"vwnd_{level}"
        if u_column in panel and v_column in panel:
            panel[f"wind_speed_{level}_mps"] = np.sqrt(panel[u_column] ** 2 + panel[v_column] ** 2)
    if {"temp_925", "temp_1000"}.issubset(panel.columns):
        panel["lower_tropospheric_stability_925_1000_c"] = panel["temp_925"] - panel["temp_1000"]
        panel["temperature_inversion_925_1000_flag"] = (
            panel["lower_tropospheric_stability_925_1000_c"] > 0
        ).astype(float)
    if {"temp_850", "temp_1000"}.issubset(panel.columns):
        panel["lower_tropospheric_stability_850_1000_c"] = panel["temp_850"] - panel["temp_1000"]
    if {"ghgt_850", "ghgt_1000", "wind_speed_850_mps"}.issubset(panel.columns):
        layer_depth = (panel["ghgt_850"] - panel["ghgt_1000"]).clip(lower=0)
        panel["ventilation_proxy_1000_850_m2_s"] = layer_depth * panel["wind_speed_850_mps"]
    value_columns = [column for column in panel if not column.startswith("n_")]
    panel["igra_feature_count"] = panel[value_columns].notna().sum(axis=1) - 3
    panel["igra_month_available"] = panel["igra_feature_count"] > 0
    return panel.sort_values(["igra_id", "year", "month"]).reset_index(drop=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_fua_month(station_month: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    station_map = candidates[
        ["igra_id", "eFUA_ID", "eFUA_name", "Cntry_ISO", "distance_to_fua_km", "recommended_upper_air_station"]
    ]
    merged = station_month.merge(station_map, on="igra_id", how="inner", validate="many_to_one")
    value_columns = [
        column
        for column in station_month.columns
        if column not in {"igra_id", "year", "month", "igra_month_available"}
        and not column.startswith("n_")
        and column != "igra_feature_count"
    ]

    def summarise(group: pd.DataFrame) -> pd.Series:
        output: dict[str, object] = {
            "igra_station_count": group["igra_id"].nunique(),
            "igra_recommended_station_count": group.loc[
                group["recommended_upper_air_station"], "igra_id"
            ].nunique(),
            "igra_source_available": group["igra_month_available"].any(),
            "nearest_igra_station_km": group["distance_to_fua_km"].min(),
            "igra_feature_count_max": group["igra_feature_count"].max(),
        }
        default_weight = pd.Series(1.0, index=group.index)
        for column in value_columns:
            count_column = f"n_{column}"
            weights = group[count_column] if count_column in group else default_weight
            output[f"upper_{column}"] = weighted_mean(group[column], weights)
        return pd.Series(output)

    panel = (
        merged.groupby(["eFUA_ID", "eFUA_name", "Cntry_ISO", "year", "month"], as_index=False)
        .apply(summarise, include_groups=False)
        .reset_index(drop=True)
    )
    return panel.sort_values(["Cntry_ISO", "eFUA_ID", "year", "month"]).reset_index(drop=True)


def main() -> int:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"igra_id": str})
    archive_records = download_archives()
    failures = [record for record in archive_records if record["status"] == "failed"]
    if failures:
        raise RuntimeError(f"Failed IGRA archives: {[record['file'] for record in failures]}")
    station_month = build_station_month(candidates)
    station_month.to_csv(STATION_MONTH_PATH, index=False, compression="gzip", encoding="utf-8", float_format="%.7g")
    fua_month = aggregate_fua_month(station_month, candidates)
    fua_month.to_csv(FUA_MONTH_PATH, index=False, compression="gzip", encoding="utf-8", float_format="%.7g")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA NCEI Integrated Global Radiosonde Archive v2.2 monthly means",
        "source_url": f"{BASE_URL}/",
        "format_url": "https://www.ncei.noaa.gov/pub/data/igra/monthly/igra2-monthly-format.txt",
        "study_period": [START_YEAR, END_YEAR],
        "pressure_levels_hpa": LEVELS,
        "minimum_observations_per_nominal_hour_mean": 10,
        "candidate_stations": int(candidates["igra_id"].nunique()),
        "station_month_rows": int(len(station_month)),
        "fua_month_rows": int(len(fua_month)),
        "fua_count": int(fua_month["eFUA_ID"].nunique()),
        "archive_records": archive_records,
        "outputs": [str(STATION_MONTH_PATH.relative_to(ROOT)), str(FUA_MONTH_PATH.relative_to(ROOT))],
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "archive_records"}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
