from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from scipy.spatial import cKDTree


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
AU_WEEKLY_PANEL = ROOT / "data/processed/modelling_panel/australia_multistate_regulatory_pm25_weekly_2015_2024.csv.gz"

FIRMS_ROOT = ROOT / "data/raw/firms_modis_country"
NPI_ROOT = ROOT / "data/raw/australia_npi/2026_snapshot"
PBLH_ROOT = ROOT / "data/raw/era5_pblh_open_meteo"
OUT_ROOT = ROOT / "data/processed/dynamic_sources"
META_ROOT = ROOT / "data/metadata"

FIRE_MONTHLY_OUT = OUT_ROOT / "fua_month_modis_fire_2000_2025.csv.gz"
FIRE_WEEKLY_OUT = OUT_ROOT / "australia_fua_week_modis_fire_2015_2025.csv.gz"
PBLH_MONTHLY_OUT = OUT_ROOT / "australia_monitor_fua_month_era5_pblh_2000_2025.csv.gz"
PBLH_WEEKLY_OUT = OUT_ROOT / "australia_monitor_fua_week_era5_pblh_2015_2025.csv.gz"
NPI_OUT = OUT_ROOT / "australia_fua_year_npi_emissions.csv.gz"
AUDIT_OUT = META_ROOT / "dynamic_source_extensions_20260829.json"

EARTH_RADIUS_KM = 6371.0088
FIRMS_COUNTRY = {"CHN": "China", "AUS": "Australia"}
FIRMS_START = pd.Timestamp("2000-11-01")
FIRE_RADII_KM = (50, 100, 250)
PBLH_API = "https://archive-api.open-meteo.com/v1/archive"
NPI_EMISSIONS_URL = (
    "https://data.gov.au/data/dataset/043f58e0-a188-4458-b61c-04e5b540aea4/"
    "resource/9ec26edf-ea5b-4869-8c2c-38b41fb3a51d/download/emissions.csv"
)
NPI_SUBSTANCES = {
    "Particulate Matter ≤2.5 µm (PM2.5)": "pm25",
    "Oxides of Nitrogen": "nox",
    "Sulfur dioxide": "so2",
    "Ammonia (total)": "nh3",
    "Total Volatile Organic Compounds": "voc",
}


def ensure_dirs() -> None:
    for path in (FIRMS_ROOT, NPI_ROOT, PBLH_ROOT, OUT_ROOT, META_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_to_gzip(url: str, destination: Path, timeout: int = 300) -> Path:
    if destination.exists() and destination.stat().st_size > 100:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with gzip.open(temporary, "wb", compresslevel=6) as target:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    target.write(block)
    temporary.replace(destination)
    return destination


def download_plain(url: str, destination: Path, timeout: int = 600) -> Path:
    if destination.exists() and destination.stat().st_size > 100:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with temporary.open("wb") as target:
            for block in response.iter_content(chunk_size=1024 * 1024):
                if block:
                    target.write(block)
    temporary.replace(destination)
    return destination


def load_fuas() -> pd.DataFrame:
    columns = [
        "eFUA_ID",
        "fua_name",
        "country_iso",
        "fua_centroid_lon",
        "fua_centroid_lat",
    ]
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(PANEL, usecols=columns, chunksize=150_000, low_memory=False):
        parts.append(chunk.drop_duplicates("eFUA_ID"))
    frame = pd.concat(parts, ignore_index=True).drop_duplicates("eFUA_ID")
    frame["eFUA_ID"] = pd.to_numeric(frame["eFUA_ID"], errors="raise").astype(int)
    frame["fua_centroid_lon"] = pd.to_numeric(frame["fua_centroid_lon"], errors="raise")
    frame["fua_centroid_lat"] = pd.to_numeric(frame["fua_centroid_lat"], errors="raise")
    frame = frame[frame["country_iso"].isin(FIRMS_COUNTRY)].copy()
    return frame.sort_values(["country_iso", "eFUA_ID"]).reset_index(drop=True)


def unit_sphere(longitude: Iterable[float], latitude: Iterable[float]) -> np.ndarray:
    lon = np.radians(np.asarray(longitude, dtype=float))
    lat = np.radians(np.asarray(latitude, dtype=float))
    cos_lat = np.cos(lat)
    return np.column_stack((cos_lat * np.cos(lon), cos_lat * np.sin(lon), np.sin(lat)))


def chord_to_km(chord: np.ndarray | float) -> np.ndarray:
    value = np.clip(np.asarray(chord, dtype=float) / 2.0, 0.0, 1.0)
    return 2.0 * EARTH_RADIUS_KM * np.arcsin(value)


def km_to_chord(distance_km: float) -> float:
    return 2.0 * math.sin(distance_km / (2.0 * EARTH_RADIUS_KM))


def fire_source_path(country_name: str, year: int) -> Path:
    return FIRMS_ROOT / country_name.lower() / f"modis_{year}_{country_name}.csv.gz"


def acquire_fire_files(start_year: int = 2000, end_year: int = 2025) -> list[Path]:
    tasks: list[tuple[str, Path]] = []
    for country_name in FIRMS_COUNTRY.values():
        for year in range(start_year, end_year + 1):
            url = (
                "https://firms.modaps.eosdis.nasa.gov/data/country/"
                f"modis_{year}_{country_name}.csv"
            )
            tasks.append((url, fire_source_path(country_name, year)))
    paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(download_to_gzip, url, path): (url, path)
            for url, path in tasks
        }
        for future in as_completed(futures):
            url, path = futures[future]
            try:
                paths.append(future.result())
                print(f"fire source ready: {path.name}", flush=True)
            except requests.HTTPError as error:
                if path.name.startswith("modis_2025_") and error.response is not None and error.response.status_code == 404:
                    print(f"optional 2025 FIRMS source unavailable: {url}", flush=True)
                    continue
                raise
    return sorted(paths)


def clean_fire(path: Path) -> pd.DataFrame:
    usecols = [
        "latitude",
        "longitude",
        "acq_date",
        "confidence",
        "frp",
        "type",
    ]
    frame = pd.read_csv(path, usecols=usecols, low_memory=False)
    for column in ("latitude", "longitude", "confidence", "frp", "type"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["acq_date"] = pd.to_datetime(frame["acq_date"], errors="coerce")
    frame = frame.dropna(subset=["latitude", "longitude", "acq_date"])
    frame = frame[frame["type"].eq(0)].copy()
    frame["frp"] = frame["frp"].clip(lower=0)
    return frame


def aggregate_fire_period(
    fire: pd.DataFrame,
    fuas: pd.DataFrame,
    period_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fua_xyz = unit_sphere(fuas["fua_centroid_lon"], fuas["fua_centroid_lat"])
    for period, group in fire.groupby(period_column, observed=True):
        tree = cKDTree(unit_sphere(group["longitude"], group["latitude"]))
        nearest_chord, _ = tree.query(fua_xyz, k=1)
        base = [
            {
                "eFUA_ID": int(fua_id),
                period_column: period,
                "fire_nearest_distance_km": float(distance),
            }
            for fua_id, distance in zip(fuas["eFUA_ID"], chord_to_km(nearest_chord))
        ]
        for radius in FIRE_RADII_KM:
            neighbours = tree.query_ball_point(fua_xyz, r=km_to_chord(radius))
            frp = group["frp"].to_numpy(float)
            confidence = group["confidence"].to_numpy(float)
            for record, indices in zip(base, neighbours):
                selected = np.asarray(indices, dtype=int)
                record[f"fire_detection_count_{radius}km"] = int(len(selected))
                record[f"fire_frp_sum_{radius}km"] = float(np.nansum(frp[selected])) if len(selected) else 0.0
                if radius == 100:
                    record["fire_nominal_confidence_count_100km"] = (
                        int(np.count_nonzero(confidence[selected] >= 30.0)) if len(selected) else 0
                    )
                    record["fire_high_confidence_count_100km"] = (
                        int(np.count_nonzero(confidence[selected] >= 80.0)) if len(selected) else 0
                    )
        rows.extend(base)
    return pd.DataFrame(rows)


def build_fire_features(fuas: pd.DataFrame, paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly_parts: list[pd.DataFrame] = []
    weekly_parts: list[pd.DataFrame] = []
    available_end_year: dict[str, int] = {}
    for path in paths:
        match = re.search(r"modis_(\d{4})_(Australia|China)", path.name)
        if not match:
            raise ValueError(f"Cannot identify FIRMS country-year from {path}")
        year = int(match.group(1))
        country_name = match.group(2)
        country_iso = next(key for key, value in FIRMS_COUNTRY.items() if value == country_name)
        available_end_year[country_iso] = max(available_end_year.get(country_iso, year), year)
        country_fuas = fuas[fuas["country_iso"].eq(country_iso)].copy()
        fire = clean_fire(path)
        if not fire.empty:
            fire["year_month"] = fire["acq_date"].dt.to_period("M").dt.to_timestamp()
            monthly_parts.append(aggregate_fire_period(fire, country_fuas, "year_month"))
            if country_iso == "AUS" and year >= 2015:
                fire["week_start"] = fire["acq_date"] - pd.to_timedelta(
                    fire["acq_date"].dt.dayofweek, unit="D"
                )
                weekly_parts.append(aggregate_fire_period(fire, country_fuas, "week_start"))
        print(f"fire aggregation complete: {country_name} {year}", flush=True)

    monthly_events = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else pd.DataFrame()
    all_months = pd.DataFrame(
        {"year_month": pd.date_range("2000-01-01", "2025-12-01", freq="MS")}
    )
    monthly = fuas.assign(_key=1).merge(all_months.assign(_key=1), on="_key").drop(columns="_key")
    monthly = monthly.merge(
        monthly_events,
        on=["eFUA_ID", "year_month"],
        how="left",
        validate="one_to_one",
    )
    monthly["fire_source_available"] = monthly["year_month"].ge(FIRMS_START) & monthly[
        "year_month"
    ].dt.year.le(monthly["country_iso"].map(available_end_year).fillna(-1))
    fire_value_columns = [
        column
        for column in monthly.columns
        if column.startswith("fire_") and column not in {"fire_source_available"}
    ]
    monthly.loc[monthly["fire_source_available"], fire_value_columns] = monthly.loc[
        monthly["fire_source_available"], fire_value_columns
    ].fillna(0.0)
    monthly.loc[~monthly["fire_source_available"], fire_value_columns] = np.nan
    monthly["year"] = monthly["year_month"].dt.year.astype(int)
    monthly["month"] = monthly["year_month"].dt.month.astype(int)
    monthly = monthly.drop(columns="year_month")

    weekly_events = pd.concat(weekly_parts, ignore_index=True) if weekly_parts else pd.DataFrame()
    if not weekly_events.empty:
        aggregations: dict[str, str] = {
            "fire_nearest_distance_km": "min",
        }
        for column in weekly_events.columns:
            if column.startswith("fire_detection_count_") or column.startswith(
                "fire_nominal_confidence_count_"
            ) or column.startswith("fire_high_confidence_count_") or column.startswith(
                "fire_frp_sum_"
            ):
                aggregations[column] = "sum"
        weekly_events = (
            weekly_events.groupby(["eFUA_ID", "week_start"], observed=True)
            .agg(aggregations)
            .reset_index()
        )
    australia_fuas = fuas[fuas["country_iso"].eq("AUS")].copy()
    all_weeks = pd.DataFrame(
        {"week_start": pd.date_range("2015-01-05", "2025-12-29", freq="W-MON")}
    )
    weekly = australia_fuas.assign(_key=1).merge(all_weeks.assign(_key=1), on="_key").drop(columns="_key")
    weekly = weekly.merge(
        weekly_events,
        on=["eFUA_ID", "week_start"],
        how="left",
        validate="one_to_one",
    )
    weekly["fire_source_available"] = weekly["week_start"].dt.year.le(
        available_end_year.get("AUS", -1)
    )
    weekly_fire_columns = [
        column
        for column in weekly.columns
        if column.startswith("fire_") and column not in {"fire_source_available"}
    ]
    weekly.loc[weekly["fire_source_available"], weekly_fire_columns] = weekly.loc[
        weekly["fire_source_available"], weekly_fire_columns
    ].fillna(0.0)
    weekly.loc[~weekly["fire_source_available"], weekly_fire_columns] = np.nan

    monthly.to_csv(
        FIRE_MONTHLY_OUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    weekly.to_csv(
        FIRE_WEEKLY_OUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return monthly, weekly


def monitor_fuas(fuas: pd.DataFrame) -> pd.DataFrame:
    ids = pd.read_csv(AU_WEEKLY_PANEL, usecols=["eFUA_ID"])["eFUA_ID"]
    selected = set(pd.to_numeric(ids, errors="coerce").dropna().astype(int))
    return fuas[fuas["eFUA_ID"].isin(selected)].copy().reset_index(drop=True)


def pblh_cache_path(batch_id: int, start_year: int, end_year: int) -> Path:
    return PBLH_ROOT / f"batch_{batch_id:04d}_{start_year}_{end_year}.json.gz"


def fetch_pblh_batch(
    batch: pd.DataFrame,
    batch_id: int,
    start_year: int,
    end_year: int,
    retries: int = 5,
) -> tuple[pd.DataFrame, Path]:
    destination = pblh_cache_path(batch_id, start_year, end_year)
    if destination.exists() and destination.stat().st_size > 100:
        with gzip.open(destination, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        params = {
            "latitude": ",".join(f"{value:.5f}" for value in batch["fua_centroid_lat"]),
            "longitude": ",".join(f"{value:.5f}" for value in batch["fua_centroid_lon"]),
            "start_date": f"{start_year}-01-01",
            "end_date": f"{end_year}-12-31",
            "hourly": "boundary_layer_height",
            "models": "era5",
            "temporal_resolution": "hourly_6",
            "timezone": "UTC",
        }
        for attempt in range(retries):
            try:
                response = requests.get(PBLH_API, params=params, timeout=300)
                response.raise_for_status()
                payload = response.json()
                with gzip.open(destination, "wt", encoding="utf-8", compresslevel=6) as handle:
                    json.dump(payload, handle, separators=(",", ":"))
                break
            except (requests.RequestException, ValueError):
                if attempt + 1 == retries:
                    raise
                time.sleep(2 ** attempt)
    records = payload if isinstance(payload, list) else [payload]
    if len(records) != len(batch):
        raise ValueError(f"PBLH API returned {len(records)} locations for batch size {len(batch)}")
    rows: list[pd.DataFrame] = []
    for (_, fua), record in zip(batch.iterrows(), records):
        hourly = record.get("hourly") or {}
        times = pd.to_datetime(hourly.get("time", []), errors="coerce", utc=True)
        values = pd.to_numeric(pd.Series(hourly.get("boundary_layer_height", [])), errors="coerce")
        if len(times) != len(values):
            raise ValueError(f"PBLH time/value length mismatch for FUA {fua['eFUA_ID']}")
        rows.append(
            pd.DataFrame(
                {
                    "eFUA_ID": int(fua["eFUA_ID"]),
                    "timestamp_utc": times,
                    "pblh_m": values.to_numpy(float),
                }
            )
        )
    return pd.concat(rows, ignore_index=True), destination


def summarise_pblh(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    working = frame.copy()
    working["pblh_below_500m"] = np.where(
        working["pblh_m"].notna(), working["pblh_m"].lt(500.0), np.nan
    )
    summary = (
        working.groupby(keys, observed=True)
        .agg(
            era5_pblh_mean_m=("pblh_m", "mean"),
            era5_pblh_sd_m=("pblh_m", "std"),
            era5_pblh_p10_m=("pblh_m", lambda values: values.quantile(0.10)),
            era5_pblh_p90_m=("pblh_m", lambda values: values.quantile(0.90)),
            era5_pblh_below_500m_fraction=("pblh_below_500m", "mean"),
            era5_pblh_valid_samples=("pblh_m", "count"),
        )
        .reset_index()
    )
    summary["era5_pblh_source_available"] = summary["era5_pblh_valid_samples"].gt(0)
    return summary


def build_pblh_features(fuas: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    selected = monitor_fuas(fuas)
    batches = [selected.iloc[start : start + 8].copy() for start in range(0, len(selected), 8)]
    hourly_parts: list[pd.DataFrame] = []
    source_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(fetch_pblh_batch, batch, batch_id, 2000, 2025): batch_id
            for batch_id, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            hourly, source = future.result()
            hourly_parts.append(hourly)
            source_paths.append(source)
            print(f"PBLH batch complete: {futures[future]}", flush=True)
    hourly = pd.concat(hourly_parts, ignore_index=True)
    hourly["year"] = hourly["timestamp_utc"].dt.year.astype(int)
    hourly["month"] = hourly["timestamp_utc"].dt.month.astype(int)
    hourly["week_start"] = (
        hourly["timestamp_utc"].dt.tz_localize(None).dt.normalize()
        - pd.to_timedelta(hourly["timestamp_utc"].dt.dayofweek, unit="D")
    )
    monthly = summarise_pblh(hourly, ["eFUA_ID", "year", "month"])
    weekly = summarise_pblh(
        hourly[hourly["year"].between(2015, 2025)], ["eFUA_ID", "week_start"]
    )
    monthly.to_csv(
        PBLH_MONTHLY_OUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    weekly.to_csv(
        PBLH_WEEKLY_OUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return monthly, weekly, sorted(source_paths)


def report_end_year(value: Any) -> float:
    years = [int(item) for item in re.findall(r"(?:19|20)\d{2}", str(value))]
    return float(max(years)) if years else np.nan


def build_npi_features(fuas: pd.DataFrame) -> tuple[pd.DataFrame, Path]:
    source = download_plain(NPI_EMISSIONS_URL, NPI_ROOT / "emissions.csv")
    selected_parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source, chunksize=250_000, low_memory=False):
        chunk.columns = [str(column).strip().lower() for column in chunk.columns]
        required = {
            "report_year",
            "facility_id",
            "substance_name",
            "air_total_emission_kg",
            "latitude",
            "longitude",
        }
        if not required.issubset(chunk.columns):
            missing = sorted(required.difference(chunk.columns))
            raise ValueError(f"NPI emissions schema missing columns: {missing}")
        chunk = chunk[chunk["substance_name"].isin(NPI_SUBSTANCES)].copy()
        if not chunk.empty:
            selected_parts.append(chunk[list(required)])
    selected = pd.concat(selected_parts, ignore_index=True)
    selected["report_end_year"] = selected["report_year"].map(report_end_year)
    for column in ("air_total_emission_kg", "latitude", "longitude"):
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    selected = selected.dropna(
        subset=["report_end_year", "facility_id", "latitude", "longitude"]
    )
    selected["report_end_year"] = selected["report_end_year"].astype(int)
    facilities = (
        selected.groupby(
            [
                "report_end_year",
                "facility_id",
                "substance_name",
                "latitude",
                "longitude",
            ],
            observed=True,
        )["air_total_emission_kg"]
        .sum(min_count=1)
        .reset_index()
    )
    australia = fuas[fuas["country_iso"].eq("AUS")].copy()
    fua_xyz = unit_sphere(australia["fua_centroid_lon"], australia["fua_centroid_lat"])
    rows: list[dict[str, Any]] = []
    for report_year, year_frame in facilities.groupby("report_end_year", observed=True):
        base = [
            {"eFUA_ID": int(fua_id), "npi_report_end_year": int(report_year)}
            for fua_id in australia["eFUA_ID"]
        ]
        for substance, substance_frame in year_frame.groupby("substance_name", observed=True):
            short_name = NPI_SUBSTANCES[str(substance)]
            tree = cKDTree(unit_sphere(substance_frame["longitude"], substance_frame["latitude"]))
            emissions = substance_frame["air_total_emission_kg"].to_numpy(float)
            facilities_count = substance_frame["facility_id"].astype(str).to_numpy()
            for radius in (50, 100):
                neighbours = tree.query_ball_point(fua_xyz, r=km_to_chord(radius))
                for record, indices in zip(base, neighbours):
                    selected_indices = np.asarray(indices, dtype=int)
                    record[f"npi_{short_name}_kg_{radius}km"] = (
                        float(np.nansum(emissions[selected_indices])) if len(selected_indices) else 0.0
                    )
                    record[f"npi_{short_name}_facility_count_{radius}km"] = (
                        int(pd.unique(facilities_count[selected_indices]).size) if len(selected_indices) else 0
                    )
        rows.extend(base)
    output = pd.DataFrame(rows).sort_values(["eFUA_ID", "npi_report_end_year"])
    output.to_csv(
        NPI_OUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    return output, source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sources",
        default="fire,pblh,npi",
        help="Comma-separated subset of fire,pblh,npi",
    )
    args = parser.parse_args()
    requested = {item.strip().lower() for item in args.sources.split(",") if item.strip()}
    unknown = requested.difference({"fire", "pblh", "npi"})
    if unknown:
        raise ValueError(f"Unknown sources: {sorted(unknown)}")

    started = time.time()
    ensure_dirs()
    fuas = load_fuas()
    source_paths: list[Path] = []
    outputs: dict[str, Any] = {}

    if "fire" in requested:
        fire_paths = acquire_fire_files()
        source_paths.extend(fire_paths)
        fire_monthly, fire_weekly = build_fire_features(fuas, fire_paths)
        outputs["fire_monthly_rows"] = len(fire_monthly)
        outputs["fire_weekly_rows"] = len(fire_weekly)
        outputs["fire_fuas"] = int(fire_monthly["eFUA_ID"].nunique())

    if "pblh" in requested:
        pblh_monthly, pblh_weekly, pblh_paths = build_pblh_features(fuas)
        source_paths.extend(pblh_paths)
        outputs["pblh_monthly_rows"] = len(pblh_monthly)
        outputs["pblh_weekly_rows"] = len(pblh_weekly)
        outputs["pblh_fuas"] = int(pblh_monthly["eFUA_ID"].nunique())

    if "npi" in requested:
        npi, npi_source = build_npi_features(fuas)
        source_paths.append(npi_source)
        outputs["npi_rows"] = len(npi)
        outputs["npi_fuas"] = int(npi["eFUA_ID"].nunique())
        outputs["npi_years"] = sorted(npi["npi_report_end_year"].unique().astype(int).tolist())

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": time.time() - started,
        "requested_sources": sorted(requested),
        "fua_count": int(fuas["eFUA_ID"].nunique()),
        "country_fua_counts": fuas.groupby("country_iso")["eFUA_ID"].nunique().to_dict(),
        "fire_definition": {
            "instrument": "MODIS Collection 6.1 standard-processing active-fire detections",
            "type_filter": "type == 0 vegetation fires",
            "primary_buffers_km": list(FIRE_RADII_KM),
            "confidence_thresholds": {"nominal": 30, "high": 80},
            "interpretation": "hotspot detections, not unique fire events or burned area",
        },
        "pblh_definition": {
            "source": "ERA5 via Open-Meteo historical API",
            "temporal_sampling": "6-hourly",
            "scope": "Australian FUAs represented by the regulatory monitor panel",
        },
        "npi_definition": {
            "source": "Australian National Pollutant Inventory",
            "pollutants_kept_separate": list(NPI_SUBSTANCES.values()),
            "recommended_predictive_join": "one-year lag by report end year",
        },
        "outputs": outputs,
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): sha256(path)
            for path in sorted(set(source_paths))
            if path.exists()
        },
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps({"outputs": outputs, "runtime_seconds": audit["runtime_seconds"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
