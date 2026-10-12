from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


START_YEAR = 2000
END_YEAR = 2025
MIN_VALID_DAYS = 15
DRY_RH_THRESHOLD_PCT = 80.0
DOWNLOAD_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station_id}.csv"

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = ROOT / "data/interim/aviation/isd_station_candidates_2000_2025.csv"
YEAR_INDEX_DIR = ROOT / "tmp/aviation_feasibility_20260822/year_indexes"
RAW_DIR = ROOT / "data/raw/meteorology/noaa_gsod_china_australia_2000_2025"
INTERIM_DIR = ROOT / "data/interim/aviation"
META_DIR = ROOT / "data/metadata"

STATION_MONTH_PATH = INTERIM_DIR / "gsod_station_month_2000_2025.csv.gz"
FUA_MONTH_PATH = INTERIM_DIR / "gsod_fua_month_2000_2025.csv.gz"
MANIFEST_PATH = META_DIR / "gsod_download_manifest_2000_2025.csv"
METADATA_PATH = META_DIR / "gsod_monthly_surface_panel_2000_2025.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a monthly aviation/surface panel from NOAA GSOD.")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--parse-workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional station-year limit for smoke tests.")
    return parser.parse_args()


def expected_station_years(candidates: pd.DataFrame) -> list[tuple[str, int]]:
    candidate_ids = set(candidates["station_id"].astype(str).str.zfill(11))
    pairs: list[tuple[str, int]] = []
    for year in range(START_YEAR, END_YEAR + 1):
        index_path = YEAR_INDEX_DIR / f"{year}.html"
        ids = set(re.findall(r'href="(\d{11})\.csv"', index_path.read_text(encoding="utf-8")))
        pairs.extend((station_id, year) for station_id in sorted(ids & candidate_ids))
    return pairs


def raw_path(station_id: str, year: int) -> Path:
    return RAW_DIR / str(year) / f"{station_id}.csv.gz"


def validate_csv_bytes(content: bytes) -> bool:
    return content.startswith(b'"STATION","DATE"') and content.count(b"\n") >= 2


def download_one(station_id: str, year: int, retries: int, timeout: int) -> dict[str, object]:
    destination = raw_path(station_id, year)
    if destination.exists() and destination.stat().st_size > 100:
        return {
            "station_id": station_id,
            "year": year,
            "status": "existing",
            "compressed_bytes": destination.stat().st_size,
            "url": DOWNLOAD_URL.format(year=year, station_id=station_id),
        }

    destination.parent.mkdir(parents=True, exist_ok=True)
    url = DOWNLOAD_URL.format(year=year, station_id=station_id)
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 404:
                return {
                    "station_id": station_id,
                    "year": year,
                    "status": "not_found",
                    "compressed_bytes": 0,
                    "url": url,
                }
            response.raise_for_status()
            if not validate_csv_bytes(response.content):
                raise ValueError("response was not a valid non-empty GSOD CSV")
            with gzip.open(destination, "wb", compresslevel=6) as stream:
                stream.write(response.content)
            return {
                "station_id": station_id,
                "year": year,
                "status": "downloaded",
                "compressed_bytes": destination.stat().st_size,
                "url": url,
            }
        except Exception as exc:  # Network failures are recorded and retried.
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(2**attempt, 10))
    return {
        "station_id": station_id,
        "year": year,
        "status": "failed",
        "compressed_bytes": 0,
        "url": url,
        "error": last_error,
    }


def download_all(pairs: list[tuple[str, int]], workers: int, retries: int, timeout: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    total = len(pairs)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(download_one, station_id, year, retries, timeout): (station_id, year)
            for station_id, year in pairs
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            records.append(future.result())
            if completed % 500 == 0 or completed == total:
                counts = pd.Series([record["status"] for record in records]).value_counts().to_dict()
                print(f"downloads {completed:,}/{total:,}: {counts}", flush=True)
    return pd.DataFrame(records).sort_values(["year", "station_id"])


def clean_numeric(series: pd.Series, missing_at_or_above: float) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.mask(values >= missing_at_or_above)


def fahrenheit_to_celsius(series: pd.Series) -> pd.Series:
    return (clean_numeric(series, 9999.0) - 32.0) * 5.0 / 9.0


def relative_humidity(temp_c: pd.Series, dewpoint_c: pd.Series) -> pd.Series:
    exponent_dew = 17.625 * dewpoint_c / (243.04 + dewpoint_c)
    exponent_temp = 17.625 * temp_c / (243.04 + temp_c)
    rh = 100.0 * np.exp(exponent_dew - exponent_temp)
    return rh.clip(0, 100)


def harmonic_mean(values: pd.Series) -> float:
    valid = pd.to_numeric(values, errors="coerce").dropna()
    valid = valid[valid > 0]
    if valid.empty:
        return math.nan
    return float(len(valid) / np.reciprocal(valid).sum())


def quantile(series: pd.Series, probability: float) -> float:
    valid = pd.to_numeric(series, errors="coerce").dropna()
    return float(valid.quantile(probability)) if not valid.empty else math.nan


def parse_daily(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, compression="gzip", dtype={"STATION": str, "FRSHTT": str})
    frame["station_id"] = frame["STATION"].astype(str).str.zfill(11)
    frame["date"] = pd.to_datetime(frame["DATE"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    frame["year"] = frame["date"].dt.year
    frame["month"] = frame["date"].dt.month

    frame["temp_c"] = fahrenheit_to_celsius(frame["TEMP"])
    frame["dewpoint_c"] = fahrenheit_to_celsius(frame["DEWP"])
    frame["temp_max_c"] = fahrenheit_to_celsius(frame["MAX"])
    frame["temp_min_c"] = fahrenheit_to_celsius(frame["MIN"])
    frame["rh_pct"] = relative_humidity(frame["temp_c"], frame["dewpoint_c"])
    frame["slp_hpa"] = clean_numeric(frame["SLP"], 9999.0)
    frame["stp_hpa"] = clean_numeric(frame["STP"], 9999.0)
    frame["visibility_km"] = clean_numeric(frame["VISIB"], 999.0) * 1.609344
    frame["wind_mps"] = clean_numeric(frame["WDSP"], 999.0) * 0.514444
    frame["max_wind_mps"] = clean_numeric(frame["MXSPD"], 999.0) * 0.514444
    frame["gust_mps"] = clean_numeric(frame["GUST"], 999.0) * 0.514444
    frame["precip_mm"] = clean_numeric(frame["PRCP"], 99.0) * 25.4

    flags = frame["FRSHTT"].fillna("").astype(str).str.replace(r"\D", "", regex=True).str.zfill(6).str[-6:]
    for position, name in enumerate(["fog", "rain", "snow", "hail", "thunder", "tornado"]):
        frame[f"{name}_flag"] = flags.str[position].eq("1")
    frame["dry_visibility_eligible"] = (
        frame["visibility_km"].notna()
        & frame["rh_pct"].lt(DRY_RH_THRESHOLD_PCT)
        & ~frame[["fog_flag", "rain_flag", "snow_flag"]].any(axis=1)
    )
    return frame


def aggregate_station_month_vectorized(daily: pd.DataFrame) -> pd.DataFrame:
    keys = ["station_id", "year", "month"]
    daily = daily.copy()
    daily["temperature_range_c"] = daily["temp_max_c"] - daily["temp_min_c"]
    daily["dry_visibility_km"] = daily["visibility_km"].where(daily["dry_visibility_eligible"])
    daily["visibility_inverse"] = np.where(daily["visibility_km"] > 0, 1.0 / daily["visibility_km"], np.nan)
    daily["dry_visibility_inverse"] = np.where(
        daily["dry_visibility_km"] > 0, 1.0 / daily["dry_visibility_km"], np.nan
    )
    daily["low_visibility_indicator"] = np.where(
        daily["visibility_km"].notna(), daily["visibility_km"] < 10.0, np.nan
    )
    daily["stagnant_indicator"] = np.where(daily["wind_mps"].notna(), daily["wind_mps"] < 1.0, np.nan)
    daily["wet_day_indicator"] = np.where(daily["precip_mm"].notna(), daily["precip_mm"] > 0.1, np.nan)
    grouped = daily.groupby(keys, sort=False)
    monthly = grouped.agg(
        daily_record_count=("date", "nunique"),
        temperature_valid_days=("temp_c", "count"),
        visibility_valid_days=("visibility_km", "count"),
        dry_visibility_valid_days=("dry_visibility_km", "count"),
        wind_valid_days=("wind_mps", "count"),
        precipitation_valid_days=("precip_mm", "count"),
        temperature_mean_c=("temp_c", "mean"),
        temperature_sd_c=("temp_c", "std"),
        temperature_range_mean_c=("temperature_range_c", "mean"),
        dewpoint_mean_c=("dewpoint_c", "mean"),
        relative_humidity_mean_pct=("rh_pct", "mean"),
        sea_level_pressure_mean_hpa=("slp_hpa", "mean"),
        station_pressure_mean_hpa=("stp_hpa", "mean"),
        visibility_mean_km=("visibility_km", "mean"),
        visibility_median_km=("visibility_km", "median"),
        visibility_inverse_sum=("visibility_inverse", "sum"),
        dry_visibility_inverse_sum=("dry_visibility_inverse", "sum"),
        low_visibility_day_fraction=("low_visibility_indicator", "mean"),
        wind_mean_mps=("wind_mps", "mean"),
        max_wind_mean_mps=("max_wind_mps", "mean"),
        gust_mean_mps=("gust_mps", "mean"),
        stagnant_day_fraction=("stagnant_indicator", "mean"),
        wet_day_count=("wet_day_indicator", "sum"),
        fog_day_count=("fog_flag", "sum"),
        rain_day_count=("rain_flag", "sum"),
        snow_day_count=("snow_flag", "sum"),
        hail_day_count=("hail_flag", "sum"),
        thunder_day_count=("thunder_flag", "sum"),
    )
    monthly["precipitation_sum_mm"] = grouped["precip_mm"].sum(min_count=1)
    monthly["visibility_harmonic_mean_km"] = (
        monthly["visibility_valid_days"] / monthly["visibility_inverse_sum"].replace(0, np.nan)
    )
    monthly["dry_visibility_harmonic_mean_km"] = (
        monthly["dry_visibility_valid_days"] / monthly["dry_visibility_inverse_sum"].replace(0, np.nan)
    )
    monthly["temperature_p10_c"] = grouped["temp_c"].quantile(0.10)
    monthly["temperature_p90_c"] = grouped["temp_c"].quantile(0.90)
    monthly["relative_humidity_p90_pct"] = grouped["rh_pct"].quantile(0.90)
    monthly["visibility_p10_km"] = grouped["visibility_km"].quantile(0.10)
    monthly["wind_p90_mps"] = grouped["wind_mps"].quantile(0.90)
    return monthly.drop(
        columns=["visibility_inverse_sum", "dry_visibility_inverse_sum"]
    ).reset_index()


def aggregate_station_month(group: pd.DataFrame) -> pd.Series:
    dry_visibility = group.loc[group["dry_visibility_eligible"], "visibility_km"]
    visibility = group["visibility_km"]
    return pd.Series(
        {
            "daily_record_count": group["date"].nunique(),
            "temperature_valid_days": group["temp_c"].notna().sum(),
            "visibility_valid_days": visibility.notna().sum(),
            "dry_visibility_valid_days": dry_visibility.notna().sum(),
            "wind_valid_days": group["wind_mps"].notna().sum(),
            "precipitation_valid_days": group["precip_mm"].notna().sum(),
            "temperature_mean_c": group["temp_c"].mean(),
            "temperature_sd_c": group["temp_c"].std(),
            "temperature_p10_c": quantile(group["temp_c"], 0.10),
            "temperature_p90_c": quantile(group["temp_c"], 0.90),
            "temperature_range_mean_c": (group["temp_max_c"] - group["temp_min_c"]).mean(),
            "dewpoint_mean_c": group["dewpoint_c"].mean(),
            "relative_humidity_mean_pct": group["rh_pct"].mean(),
            "relative_humidity_p90_pct": quantile(group["rh_pct"], 0.90),
            "sea_level_pressure_mean_hpa": group["slp_hpa"].mean(),
            "station_pressure_mean_hpa": group["stp_hpa"].mean(),
            "visibility_mean_km": visibility.mean(),
            "visibility_median_km": visibility.median(),
            "visibility_p10_km": quantile(visibility, 0.10),
            "visibility_harmonic_mean_km": harmonic_mean(visibility),
            "dry_visibility_harmonic_mean_km": harmonic_mean(dry_visibility),
            "low_visibility_day_fraction": (visibility < 10.0).sum() / visibility.notna().sum()
            if visibility.notna().any()
            else math.nan,
            "wind_mean_mps": group["wind_mps"].mean(),
            "wind_p90_mps": quantile(group["wind_mps"], 0.90),
            "max_wind_mean_mps": group["max_wind_mps"].mean(),
            "gust_mean_mps": group["gust_mps"].mean(),
            "stagnant_day_fraction": (group["wind_mps"] < 1.0).sum() / group["wind_mps"].notna().sum()
            if group["wind_mps"].notna().any()
            else math.nan,
            "precipitation_sum_mm": group["precip_mm"].sum(min_count=1),
            "wet_day_count": (group["precip_mm"] > 0.1).sum(),
            "fog_day_count": group["fog_flag"].sum(),
            "rain_day_count": group["rain_flag"].sum(),
            "snow_day_count": group["snow_flag"].sum(),
            "hail_day_count": group["hail_flag"].sum(),
            "thunder_day_count": group["thunder_flag"].sum(),
        }
    )


def parse_station_year(path: Path) -> tuple[Path, pd.DataFrame | None, str | None]:
    try:
        daily = parse_daily(path)
        monthly = aggregate_station_month_vectorized(daily)
        return path, monthly, None
    except Exception as exc:
        return path, None, f"{type(exc).__name__}: {exc}"


def build_station_month(files: list[Path], parse_workers: int) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=parse_workers) as executor:
        futures = [executor.submit(parse_station_year, path) for path in files]
        for index, future in enumerate(as_completed(futures), start=1):
            path, monthly, error = future.result()
            if error:
                print(f"parse failure {path}: {error}", flush=True)
            elif monthly is not None:
                chunks.append(monthly)
            if index % 500 == 0 or index == len(files):
                print(f"parsed {index:,}/{len(files):,} station-year files", flush=True)
    panel = pd.concat(chunks, ignore_index=True)
    integer_columns = [column for column in panel if column.endswith("_days") or column.endswith("_count")]
    panel[integer_columns] = panel[integer_columns].fillna(0).astype(int)
    panel["station_month_eligible"] = panel["daily_record_count"] >= MIN_VALID_DAYS
    panel["dry_visibility_month_eligible"] = panel["dry_visibility_valid_days"] >= MIN_VALID_DAYS
    return panel.sort_values(["station_id", "year", "month"]).reset_index(drop=True)


def weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    mask = values.notna() & weights.notna() & weights.gt(0)
    if not mask.any():
        return math.nan
    return float(np.average(values[mask], weights=weights[mask]))


def aggregate_fua_month(group: pd.DataFrame, value_columns: list[str]) -> pd.Series:
    eligible = group[group["station_month_eligible"]]
    output: dict[str, object] = {
        "surface_station_count": group["station_id"].nunique(),
        "surface_eligible_station_count": eligible["station_id"].nunique(),
        "aviation_icao_station_count": group.loc[group["is_aviation_icao"], "station_id"].nunique(),
        "aviation_icao_eligible_station_count": eligible.loc[eligible["is_aviation_icao"], "station_id"].nunique(),
        "surface_source_available": not eligible.empty,
        "aviation_icao_source_available": eligible["is_aviation_icao"].any(),
        "nearest_surface_station_km": group["distance_to_fua_km"].min(),
        "median_surface_station_km": group["distance_to_fua_km"].median(),
        "surface_valid_station_days": eligible["daily_record_count"].sum(),
    }
    for column in value_columns:
        output[f"surface_{column}"] = weighted_mean(eligible[column], eligible["daily_record_count"])
    return pd.Series(output)


def build_fua_month(station_month: pd.DataFrame, candidates: pd.DataFrame) -> pd.DataFrame:
    station_map = candidates[
        ["station_id", "eFUA_ID", "eFUA_name", "Cntry_ISO", "is_aviation_icao", "distance_to_fua_km"]
    ].copy()
    station_map["station_id"] = station_map["station_id"].astype(str).str.zfill(11)
    merged = station_month.merge(station_map, on="station_id", how="inner", validate="many_to_one")
    excluded = {
        "station_id",
        "year",
        "month",
        "station_month_eligible",
        "dry_visibility_month_eligible",
        "daily_record_count",
    }
    value_columns = [column for column in station_month.columns if column not in excluded]
    panel = (
        merged.groupby(["eFUA_ID", "eFUA_name", "Cntry_ISO", "year", "month"], as_index=False)
        .apply(lambda group: aggregate_fua_month(group, value_columns), include_groups=False)
        .reset_index(drop=True)
    )
    count_columns = [column for column in panel if column.endswith("_count") or column.endswith("_days")]
    panel[count_columns] = panel[count_columns].fillna(0).astype(int)
    return panel.sort_values(["Cntry_ISO", "eFUA_ID", "year", "month"]).reset_index(drop=True)


def main() -> int:
    args = parse_args()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    candidates = pd.read_csv(CANDIDATES_PATH, dtype={"station_id": str})
    candidates["station_id"] = candidates["station_id"].str.zfill(11)
    pairs = expected_station_years(candidates)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"expected station-year files: {len(pairs):,}", flush=True)

    if args.skip_download:
        manifest = pd.DataFrame(
            [
                {
                    "station_id": station_id,
                    "year": year,
                    "status": "existing" if raw_path(station_id, year).exists() else "missing",
                    "compressed_bytes": raw_path(station_id, year).stat().st_size
                    if raw_path(station_id, year).exists()
                    else 0,
                }
                for station_id, year in pairs
            ]
        )
    else:
        manifest = download_all(pairs, args.workers, args.retries, args.timeout)
    manifest.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")

    files = [raw_path(station_id, year) for station_id, year in pairs if raw_path(station_id, year).exists()]
    station_month = build_station_month(files, args.parse_workers)
    station_month.to_csv(STATION_MONTH_PATH, index=False, compression="gzip", encoding="utf-8")
    fua_month = build_fua_month(station_month, candidates)
    fua_month.to_csv(FUA_MONTH_PATH, index=False, compression="gzip", encoding="utf-8")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA NCEI Global Summary of the Day, derived from Integrated Surface Hourly",
        "source_url": "https://www.ncei.noaa.gov/data/global-summary-of-the-day/",
        "study_period": [START_YEAR, END_YEAR],
        "candidate_stations": int(candidates["station_id"].nunique()),
        "expected_station_year_files": len(pairs),
        "available_station_year_files": len(files),
        "download_status_counts": manifest["status"].value_counts().to_dict(),
        "station_month_rows": int(len(station_month)),
        "fua_month_rows_with_surface_source": int(len(fua_month)),
        "fua_count_with_surface_source": int(fua_month["eFUA_ID"].nunique()),
        "minimum_valid_days": MIN_VALID_DAYS,
        "dry_visibility_rule": (
            f"visibility present; RH < {DRY_RH_THRESHOLD_PCT:.0f}%; no fog, rain/drizzle, or snow/ice flag"
        ),
        "outputs": [
            str(STATION_MONTH_PATH.relative_to(ROOT)),
            str(FUA_MONTH_PATH.relative_to(ROOT)),
            str(MANIFEST_PATH.relative_to(ROOT)),
        ],
    }
    METADATA_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
