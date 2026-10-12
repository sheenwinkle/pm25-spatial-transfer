from __future__ import annotations

import calendar
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import run_australia_ground_monitor_validation as nsw_base


ROOT = Path(__file__).resolve().parents[1]
QLD_ROOT = ROOT / "data/raw/australia_ground_monitoring/qld"
QLD_STATION_YEAR_METADATA = QLD_ROOT / "qld_air_quality_station_year_metadata_2015_2024.csv"
VIC_ROOT = ROOT / "data/raw/australia_ground_monitoring/vic/hourly_airwatch"
VIC_MONTHLY_CACHE = ROOT / "data/interim/australia_ground_monitoring/vic_validated_pm25_monthly_2015_2024.csv.gz"
AMLB_PATH = ROOT / "data/interim/amlb_dynamic/amlb_pm25_australia_gridcell_monthly_2002_2024.csv.gz"
OUTPUT = ROOT / "data/processed/modelling_panel/australia_multistate_regulatory_pm25_monthly_2015_2024.csv.gz"
AUDIT = ROOT / "data/metadata/australia_multistate_regulatory_pm25_monthly_2015_2024.json"
SOURCE_TABLE = ROOT / "data/metadata/australia_multistate_regulatory_pm25_source_summary.csv"
MATCH_TABLE = ROOT / "data/metadata/australia_multistate_station_grid_matches.csv"
UNMATCHED_TABLE = ROOT / "data/metadata/australia_multistate_unmatched_stations.csv"

TARGET = "monitor_pm25_ug_m3"
MIN_DAILY_HOURS = 18
MIN_MONTHLY_DAY_FRACTION = 0.75
MIN_PLAUSIBLE_PM25 = -10.0
MAX_PLAUSIBLE_PM25 = 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalise_name(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def qld_header_station(value: str) -> str:
    value = re.sub(r"\s*\(ug/m\^3\)\s*$", "", str(value), flags=re.IGNORECASE)
    return value.split(" (", 1)[0].strip()


def aggregate_hourly_to_daily(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["pm25"] = pd.to_numeric(frame["pm25"], errors="coerce")
    plausible = frame["pm25"].between(MIN_PLAUSIBLE_PM25, MAX_PLAUSIBLE_PM25)
    frame.loc[~plausible, "pm25"] = np.nan
    frame = frame.dropna(subset=["datetime_local", "pm25"])
    if frame.empty:
        return pd.DataFrame()
    frame["local_hour"] = frame["datetime_local"].dt.floor("h")
    hourly = (
        frame.groupby(
            [
                "site_key",
                "source_site_id",
                "station_name",
                "network_state",
                "station_longitude",
                "station_latitude",
                "local_hour",
            ],
            dropna=False,
        )
        .agg(
            pm25=("pm25", "mean"),
            instrument_method=(
                "instrument_method",
                lambda values: "+".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )
    hourly["local_date"] = hourly["local_hour"].dt.date
    daily = (
        hourly.groupby(
            [
                "site_key",
                "source_site_id",
                "station_name",
                "network_state",
                "station_longitude",
                "station_latitude",
                "local_date",
            ],
            dropna=False,
        )
        .agg(
            daily_pm25=("pm25", "mean"),
            valid_hours=("pm25", "count"),
            instrument_method=(
                "instrument_method",
                lambda values: "+".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )
    daily = daily[daily["valid_hours"].ge(MIN_DAILY_HOURS)].copy()
    daily["local_date"] = pd.to_datetime(daily["local_date"])
    return daily


def aggregate_hourly(frame: pd.DataFrame, output_frequency: str = "monthly") -> pd.DataFrame:
    daily = aggregate_hourly_to_daily(frame)
    if daily.empty or output_frequency == "daily":
        return daily
    if output_frequency != "monthly":
        raise ValueError(f"Unsupported output frequency: {output_frequency}")
    daily["year"] = pd.to_datetime(daily["local_date"]).dt.year
    daily["month"] = pd.to_datetime(daily["local_date"]).dt.month
    monthly = (
        daily.groupby(
            [
                "site_key",
                "source_site_id",
                "station_name",
                "network_state",
                "station_longitude",
                "station_latitude",
                "year",
                "month",
            ],
            dropna=False,
        )
        .agg(
            monitor_pm25_ug_m3=("daily_pm25", "mean"),
            valid_days=("daily_pm25", "count"),
            valid_hours=("valid_hours", "sum"),
            instrument_method=(
                "instrument_method",
                lambda values: "+".join(sorted(set(values.dropna().astype(str)))),
            ),
        )
        .reset_index()
    )
    monthly["calendar_days"] = [
        calendar.monthrange(int(year), int(month))[1]
        for year, month in zip(monthly["year"], monthly["month"])
    ]
    monthly["required_valid_days"] = np.ceil(
        monthly["calendar_days"] * MIN_MONTHLY_DAY_FRACTION
    ).astype(int)
    monthly["strict_monthly_valid"] = monthly["valid_days"].ge(
        monthly["required_valid_days"]
    )
    monthly["monthly_coverage_fraction"] = (
        monthly["valid_days"] / monthly["calendar_days"]
    )
    return monthly


def load_qld_hourly_monthly(
    output_frequency: str = "monthly",
    start_year: int = 2015,
    end_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(QLD_ROOT / "monitoring-sites-on-open-data.csv")
    metadata = metadata[metadata["Monitoring type"].eq("Air quality")].copy()
    metadata["name_key"] = metadata["Site name"].map(normalise_name)
    metadata = (
        metadata.sort_values(["name_key", "End date"], na_position="last")
        .drop_duplicates("name_key", keep="last")
        .rename(
            columns={
                "Site name": "station_name",
                "Longitude": "station_longitude",
                "Latitude": "station_latitude",
            }
        )
    )
    lookup = metadata.set_index("name_key")
    station_year = pd.read_csv(QLD_STATION_YEAR_METADATA)
    station_year["name_key"] = station_year["station_name"].map(normalise_name)
    station_year = station_year.sort_values("resource_name").drop_duplicates(
        ["year", "name_key"], keep="last"
    )
    station_year_lookup = station_year.set_index(["year", "name_key"])
    parts: list[pd.DataFrame] = []
    unmatched: list[dict[str, str | int]] = []
    for year in range(start_year, end_year + 1):
        path = QLD_ROOT / "hourly_pm25" / f"pm2-5-qld-{year}.csv"
        raw = pd.read_csv(path, low_memory=False)
        if not {"Date", "Time"}.issubset(raw.columns):
            raise ValueError(f"Unexpected Queensland schema: {path}")
        utc = pd.to_datetime(
            raw["Date"].astype(str).str.strip() + " " + raw["Time"].astype(str).str.strip(),
            dayfirst=True,
            errors="coerce",
            utc=True,
        )
        local = utc.dt.tz_convert("Australia/Brisbane")
        for column in raw.columns[2:]:
            station_name = qld_header_station(column)
            name_key = normalise_name(station_name)
            station = None
            if (year, name_key) in station_year_lookup.index:
                station = station_year_lookup.loc[(year, name_key)]
            elif name_key in set(station_year["name_key"]):
                candidates = station_year[station_year["name_key"].eq(name_key)].copy()
                candidates["year_distance"] = (candidates["year"] - year).abs()
                station = candidates.sort_values(["year_distance", "year"]).iloc[0]
            elif name_key in lookup.index:
                station = lookup.loc[name_key]
            if station is None:
                unmatched.append({"network_state": "QLD", "year": year, "station_name": station_name})
                continue
            if isinstance(station, pd.DataFrame):
                station = station.iloc[0]
            hourly = pd.DataFrame(
                {
                    "site_key": f"QLD_{name_key}",
                    "source_site_id": name_key,
                    "station_name": str(station["station_name"]),
                    "network_state": "QLD",
                    "station_longitude": float(station["station_longitude"]),
                    "station_latitude": float(station["station_latitude"]),
                    "instrument_method": "QLD_validated_hourly_PM2.5",
                    "datetime_local": local,
                    "pm25": raw[column],
                }
            )
            observations = aggregate_hourly(hourly, output_frequency=output_frequency)
            if len(observations):
                parts.append(observations)
    return pd.concat(parts, ignore_index=True), pd.DataFrame(unmatched).drop_duplicates()


def victoria_location_metadata(path: Path) -> pd.DataFrame:
    metadata = pd.read_excel(path, sheet_name="Metadata", header=None)
    candidates = metadata.index[metadata.iloc[:, 0].astype(str).str.strip().eq("location_id")]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify Victoria location table in {path}")
    start = int(candidates[0])
    locations = metadata.iloc[start + 1 :, :4].copy()
    locations.columns = ["source_site_id", "station_name", "station_latitude", "station_longitude"]
    locations["source_site_id"] = pd.to_numeric(locations["source_site_id"], errors="coerce")
    locations["station_latitude"] = pd.to_numeric(locations["station_latitude"], errors="coerce")
    locations["station_longitude"] = pd.to_numeric(locations["station_longitude"], errors="coerce")
    return locations.dropna(
        subset=["source_site_id", "station_latitude", "station_longitude"]
    ).drop_duplicates("source_site_id")


def load_victoria_legacy_year(
    path: Path, year: int, sheet: str, output_frequency: str = "monthly"
) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=sheet)
    required = {
        "sample_point_id",
        "sp_name",
        "latitude",
        "longitude",
        "sample_datetime",
        "time_basis_id",
        "param_id",
        "value",
    }
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected legacy Victoria schema: {path}")
    pm25 = raw[
        raw["time_basis_id"].astype(str).str.upper().eq("1HR_AV")
        & raw["param_id"].astype(str).str.upper().isin({"BPM2.5", "SBPM25", "PM2.5"})
    ].copy()
    if pm25.empty:
        return pd.DataFrame()
    local = pd.to_datetime(pm25["sample_datetime"], errors="coerce")
    local = local.dt.tz_localize(
        "Australia/Melbourne", ambiguous="NaT", nonexistent="shift_forward"
    )
    source_id = pd.to_numeric(pm25["sample_point_id"], errors="coerce").astype("Int64")
    hourly = pd.DataFrame(
        {
            "site_key": "VIC_" + source_id.astype(str),
            "source_site_id": source_id.astype(str),
            "station_name": pm25["sp_name"].astype(str),
            "network_state": "VIC",
            "station_longitude": pd.to_numeric(pm25["longitude"], errors="coerce"),
            "station_latitude": pd.to_numeric(pm25["latitude"], errors="coerce"),
            "instrument_method": pm25["param_id"].astype(str),
            "datetime_local": local,
            "pm25": pm25["value"],
        }
    ).dropna(subset=["station_longitude", "station_latitude"])
    return aggregate_hourly(hourly, output_frequency=output_frequency)


def load_victoria_hourly_monthly(
    output_frequency: str = "monthly",
    start_year: int = 2015,
    end_year: int = 2024,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if (
        output_frequency == "monthly"
        and start_year == 2015
        and end_year == 2024
        and VIC_MONTHLY_CACHE.exists()
    ):
        return pd.read_csv(VIC_MONTHLY_CACHE, low_memory=False), pd.DataFrame()
    parts: list[pd.DataFrame] = []
    unmatched: list[dict[str, str | int]] = []
    for year in range(start_year, end_year + 1):
        path = VIC_ROOT / f"{year}_All_sites_air_quality_hourly_avg.xlsx"
        excel = pd.ExcelFile(path)
        if "Metadata" not in excel.sheet_names:
            data_sheets = [sheet for sheet in excel.sheet_names if sheet != "Data Info Panel"]
            if len(data_sheets) != 1:
                raise ValueError(f"Cannot identify legacy Victoria data sheet: {path}")
            observations = load_victoria_legacy_year(
                path, year, data_sheets[0], output_frequency=output_frequency
            )
            if len(observations):
                parts.append(observations)
            continue
        locations = victoria_location_metadata(path).set_index("source_site_id")
        for sheet in excel.sheet_names:
            if sheet in {"Data Info Panel", "Metadata", "AllData"}:
                continue
            raw = pd.read_excel(excel, sheet_name=sheet)
            if raw.empty or "location_id" not in raw.columns:
                continue
            source_id_values = pd.to_numeric(raw["location_id"], errors="coerce").dropna()
            if source_id_values.empty:
                continue
            source_id = int(source_id_values.iloc[0])
            if source_id not in locations.index:
                unmatched.append({"network_state": "VIC", "year": year, "station_name": sheet})
                continue
            station = locations.loc[source_id]
            candidates = [column for column in ("SBPM25", "BPM2.5", "PM2.5") if column in raw.columns]
            if not candidates:
                continue
            value = pd.Series(np.nan, index=raw.index, dtype=float)
            method = pd.Series("", index=raw.index, dtype=object)
            for column in candidates:
                current = pd.to_numeric(raw[column], errors="coerce")
                use = value.isna() & current.notna()
                value.loc[use] = current.loc[use]
                method.loc[use] = column
            datetime_column = "datetime_local" if "datetime_local" in raw.columns else "datetime_AEST"
            local = pd.to_datetime(raw[datetime_column], errors="coerce")
            if getattr(local.dt, "tz", None) is None:
                local = local.dt.tz_localize(
                    "Australia/Melbourne", ambiguous="NaT", nonexistent="shift_forward"
                )
            hourly = pd.DataFrame(
                {
                    "site_key": f"VIC_{source_id}",
                    "source_site_id": str(source_id),
                    "station_name": str(station["station_name"]),
                    "network_state": "VIC",
                    "station_longitude": float(station["station_longitude"]),
                    "station_latitude": float(station["station_latitude"]),
                    "instrument_method": method.replace("", np.nan),
                    "datetime_local": local,
                    "pm25": value,
                }
            ).dropna(subset=["instrument_method"])
            observations = aggregate_hourly(hourly, output_frequency=output_frequency)
            if len(observations):
                parts.append(observations)
    combined = pd.concat(parts, ignore_index=True)
    if output_frequency == "monthly" and start_year == 2015 and end_year == 2024:
        VIC_MONTHLY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(VIC_MONTHLY_CACHE, index=False, compression="gzip")
    return combined, pd.DataFrame(unmatched).drop_duplicates()


def load_nsw_monthly(proxy: pd.DataFrame) -> pd.DataFrame:
    panel, _ = nsw_base.load_nsw_monthly_panel(proxy)
    panel = panel.copy()
    panel["source_site_id"] = panel["site_id"].astype(str)
    panel["site_key"] = "NSW_" + panel["source_site_id"]
    panel["network_state"] = "NSW"
    panel["instrument_method"] = "NSW_official_monthly_API"
    panel["valid_days"] = np.nan
    panel["valid_hours"] = np.nan
    panel["calendar_days"] = [
        calendar.monthrange(int(year), int(month))[1]
        for year, month in zip(panel["year"], panel["month"])
    ]
    panel["required_valid_days"] = np.nan
    panel["strict_monthly_valid"] = True
    panel["monthly_coverage_fraction"] = np.nan
    return panel


def match_and_merge(
    observations: pd.DataFrame, proxy: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    stations = observations[
        [
            "site_key",
            "source_site_id",
            "station_name",
            "network_state",
            "station_longitude",
            "station_latitude",
        ]
    ].drop_duplicates("site_key")
    matches = nsw_base.nearest_grid_matches(stations, proxy, "site_key")
    eligible = matches[
        matches["nearest_grid_distance_km"].le(nsw_base.MAX_MATCH_DISTANCE_KM)
    ]
    panel = observations.merge(
        eligible[
            [
                "site_key",
                "cell_id",
                "eFUA_ID",
                "fua_name",
                "grid_lon",
                "grid_lat",
                "nearest_grid_distance_km",
            ]
        ],
        on="site_key",
        how="inner",
        validate="many_to_one",
    ).merge(
        proxy,
        on=["cell_id", "eFUA_ID", "fua_name", "grid_lon", "grid_lat", "year", "month"],
        how="inner",
        validate="many_to_one",
        suffixes=("", "_proxy"),
    )
    panel["monitor_minus_proxy"] = panel[TARGET] - panel[nsw_base.PROXY]
    return panel, matches


def add_engineered_predictors(panel: pd.DataFrame) -> pd.DataFrame:
    additions: dict[str, pd.Series] = {}

    def add(name: str, values: pd.Series | np.ndarray) -> None:
        additions[name] = pd.Series(values, index=panel.index).replace([np.inf, -np.inf], np.nan)

    if AMLB_PATH.exists():
        amlb = pd.read_csv(
            AMLB_PATH, usecols=["cell_id", "year", "month", "amlb_pm25_ug_m3"]
        )
        panel = panel.merge(
            amlb,
            on=["cell_id", "year", "month"],
            how="left",
            validate="many_to_one",
        )
    add("proxy_log1p", np.log1p(panel[nsw_base.PROXY].clip(lower=0)))
    add("station_distance_log1p", np.log1p(panel["nearest_grid_distance_km"].clip(lower=0)))
    add("year_centered_2015", panel["year"] - 2015)
    if {"power_t2m_max_c", "power_t2m_min_c"}.issubset(panel):
        add("power_temperature_range_c", panel["power_t2m_max_c"] - panel["power_t2m_min_c"])
    if {"surface_temperature_mean_c", "surface_dewpoint_mean_c"}.issubset(panel):
        add(
            "surface_dewpoint_depression_c",
            panel["surface_temperature_mean_c"] - panel["surface_dewpoint_mean_c"],
        )
    for source in ("surface_wind_mean_mps", "power_ws10m_m_s"):
        if source in panel:
            add(f"inverse_{source}", 1.0 / (panel[source].clip(lower=0) + 0.25))
    for source in (
        "surface_dry_visibility_harmonic_mean_km",
        "surface_visibility_harmonic_mean_km",
    ):
        if source in panel:
            add(f"extinction_{source}", 1.0 / (panel[source].clip(lower=0) + 0.5))
    for source in ("surface_precipitation_sum_mm", "power_prectotcorr_mm_day"):
        if source in panel:
            add(f"log1p_{source}", np.log1p(panel[source].clip(lower=0)))
    if {"surface_stagnant_day_fraction", "surface_relative_humidity_mean_pct"}.issubset(panel):
        add(
            "stagnation_humidity_interaction",
            panel["surface_stagnant_day_fraction"]
            * panel["surface_relative_humidity_mean_pct"]
            / 100.0,
        )
    if {"surface_wind_mean_mps", "surface_precipitation_sum_mm"}.issubset(panel):
        add(
            "wind_wet_scavenging_interaction",
            panel["surface_wind_mean_mps"]
            * np.log1p(panel["surface_precipitation_sum_mm"].clip(lower=0)),
        )
    emission_columns = [
        column
        for column in panel.columns
        if column.startswith("edgar_") and column.endswith("_tonnes_gridcell")
    ]
    for column in emission_columns:
        add(f"log1p_{column}", np.log1p(panel[column].clip(lower=0)))
    for column in (
        "fua_population_2015",
        "urban_centre_population_2015",
        "population_density_2015_per_km2",
    ):
        if column in panel:
            add(f"log1p_{column}", np.log1p(panel[column].clip(lower=0)))
    return pd.concat([panel.copy(), pd.DataFrame(additions)], axis=1)


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    proxy, _ = nsw_base.load_australia_proxy_panel()
    nsw_panel = load_nsw_monthly(proxy)
    qld_observations, qld_unmatched = load_qld_hourly_monthly()
    vic_observations, vic_unmatched = load_victoria_hourly_monthly()
    qld_panel, qld_matches = match_and_merge(qld_observations, proxy)
    vic_panel, vic_matches = match_and_merge(vic_observations, proxy)

    common = pd.concat([nsw_panel, qld_panel, vic_panel], ignore_index=True, sort=False)
    common = add_engineered_predictors(common)
    common = common.sort_values(["site_key", "year", "month"]).reset_index(drop=True)
    if common.duplicated(["site_key", "year", "month"]).any():
        duplicates = common[common.duplicated(["site_key", "year", "month"], keep=False)]
        raise ValueError(f"Duplicate multi-state station-month keys: {len(duplicates)}")
    common.to_csv(
        OUTPUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )

    matches = pd.concat(
        [
            nsw_panel[
                [
                    "site_key",
                    "source_site_id",
                    "station_name",
                    "network_state",
                    "station_longitude",
                    "station_latitude",
                    "cell_id",
                    "eFUA_ID",
                    "fua_name",
                    "grid_lon",
                    "grid_lat",
                    "nearest_grid_distance_km",
                ]
            ].drop_duplicates("site_key"),
            qld_matches,
            vic_matches,
        ],
        ignore_index=True,
        sort=False,
    ).drop_duplicates("site_key")
    matches.to_csv(MATCH_TABLE, index=False)
    unmatched = pd.concat([qld_unmatched, vic_unmatched], ignore_index=True, sort=False)
    unmatched.to_csv(UNMATCHED_TABLE, index=False)

    summary = (
        common.groupby("network_state")
        .agg(
            station_months=(TARGET, "size"),
            strict_station_months=("strict_monthly_valid", "sum"),
            stations=("site_key", "nunique"),
            fuas=("fua_name", "nunique"),
            first_year=("year", "min"),
            last_year=("year", "max"),
            target_mean=(TARGET, "mean"),
            target_sd=(TARGET, "std"),
        )
        .reset_index()
    )
    summary.to_csv(SOURCE_TABLE, index=False)
    sources = list((QLD_ROOT / "hourly_pm25").glob("*.csv")) + list(
        VIC_ROOT.glob("*.xlsx")
    )
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
        "rows": len(common),
        "strict_rows": int(common["strict_monthly_valid"].sum()),
        "stations": int(common["site_key"].nunique()),
        "states": sorted(common["network_state"].unique().tolist()),
        "fuas": sorted(common["fua_name"].unique().tolist()),
        "quality_rule": {
            "minimum_valid_hours_per_day": MIN_DAILY_HOURS,
            "minimum_valid_day_fraction_per_month": MIN_MONTHLY_DAY_FRACTION,
            "plausible_hourly_range_ug_m3": [MIN_PLAUSIBLE_PM25, MAX_PLAUSIBLE_PM25],
            "target_based_test_trimming": False,
        },
        "amlb_australia_coverage": float(common.get("amlb_pm25_ug_m3", pd.Series(dtype=float)).notna().mean()),
        "source_files": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in sorted(sources)
        },
        "source_summary": summary.to_dict(orient="records"),
        "unmatched_source_station_names": int(len(unmatched)),
    }
    AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(json.dumps({key: audit[key] for key in ("rows", "strict_rows", "stations", "states", "fuas", "amlb_australia_coverage")}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
