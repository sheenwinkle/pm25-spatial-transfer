from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
OUTPUT = ROOT / "data/metadata/monthly_multisource_data_dictionary_2000_2024.csv"


IDENTIFIERS = {
    "cell_id": "Stable 0.1-degree study-grid identifier",
    "eFUA_ID": "GHS Functional Urban Area identifier",
    "fua_name": "Functional Urban Area name",
    "country_iso": "ISO alpha-3 country code",
    "country_name": "Country name",
    "year": "Calendar year",
    "month": "Calendar month number",
    "date": "First day of the represented calendar month",
}


def source_group(column: str) -> str:
    if column in IDENTIFIERS:
        return "identifier"
    if column == "pm25_mean_ug_m3" or column.startswith("pm25_"):
        return "SatPM target/history"
    if column.startswith("edgar_"):
        return "EDGAR v8.1"
    if column.startswith("power_"):
        return "NASA POWER"
    if column.startswith("surface_") or column.startswith("aviation_icao_") or column in {
        "nearest_surface_station_km",
        "median_surface_station_km",
    }:
        return "NOAA GSOD/ISD surface"
    if column.startswith("upper_") or column.startswith("igra_") or column == "nearest_igra_station_km":
        return "NOAA IGRA upper air"
    if column.startswith("annual_covariate_"):
        return "temporal provenance"
    if column in {"month_sin", "month_cos", "time_index_month"}:
        return "derived time"
    if any(token in column for token in ["population", "urban_centre", "commuting_zone", "fua_area"]):
        return "GHS-FUA urban form"
    if any(token in column for token in ["lon", "lat", "coast", "coastal", "cell_source"]):
        return "derived geography"
    return "derived/core"


def role(column: str) -> str:
    if column in IDENTIFIERS:
        return "identifier"
    if column == "pm25_mean_ug_m3":
        return "target"
    if column.endswith("available") or column.endswith("flag") or column.endswith("missing_flag"):
        return "availability/quality flag"
    if column.startswith("pm25_lag") or column.startswith("pm25_roll") or column.startswith("pm25_change"):
        return "forecast-history feature"
    if column.startswith("annual_covariate_"):
        return "provenance"
    return "predictor"


def unit(column: str) -> str:
    rules = [
        ("pm25", "µg/m³"),
        ("_km2", "km²"),
        ("_km", "km"),
        ("_mps", "m/s"),
        ("_m_s", "m/s"),
        ("_hpa", "hPa"),
        ("_pct", "%"),
        ("percent", "%"),
        ("_c", "°C"),
        ("_mm", "mm"),
        ("mm_day", "mm/day"),
        ("tonnes", "tonnes/grid cell/year"),
        ("population", "persons"),
        ("density", "persons/km²"),
        ("_lon", "decimal degrees"),
        ("_lat", "decimal degrees"),
        ("ghgt", "geopotential metres"),
        ("vapr", "Pa (archive monthly mean)"),
        ("_days", "days"),
        ("_count", "count"),
        ("fraction", "proportion"),
        ("share", "proportion"),
    ]
    for token, value in rules:
        if token in column:
            return value
    return "unitless/not applicable"


def description(column: str) -> str:
    if column in IDENTIFIERS:
        return IDENTIFIERS[column]
    if column == "pm25_mean_ug_m3":
        return "Monthly SatPM V6.GL.03 PM2.5 value sampled at the study grid cell"
    text = column.replace("_", " ")
    if column.startswith("surface_"):
        return "FUA-month station-support-weighted surface statistic: " + text.removeprefix("surface ")
    if column.startswith("upper_"):
        return "FUA-month radiosonde-derived upper-air statistic: " + text.removeprefix("upper ")
    if column.startswith("edgar_"):
        return "Annual EDGAR grid-cell emissions predictor repeated within year: " + text.removeprefix("edgar ")
    if column.startswith("power_"):
        return "Annual NASA POWER predictor repeated within year: " + text.removeprefix("power ")
    if column.startswith("pm25_lag"):
        return "PM2.5 value observed at the stated prior-month lag within grid cell"
    if column.startswith("pm25_roll"):
        return "Lagged rolling PM2.5 mean using only months before the target month"
    if column.startswith("pm25_change"):
        return "Lagged monthly PM2.5 change using only months before the target month"
    if column.endswith("missing_flag"):
        return "Indicator that the corresponding source-availability field was absent before defaulting"
    return text.capitalize()


def main() -> int:
    columns = pd.read_csv(PANEL, nrows=0).columns.tolist()
    missing = pd.Series(0, index=columns, dtype="int64")
    dtypes: dict[str, str] = {}
    rows = 0
    for chunk in pd.read_csv(PANEL, chunksize=100_000, low_memory=False):
        rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")
        for column, dtype in chunk.dtypes.items():
            dtypes.setdefault(column, str(dtype))
    dictionary = pd.DataFrame(
        {
            "column": columns,
            "source_group": [source_group(column) for column in columns],
            "role": [role(column) for column in columns],
            "description": [description(column) for column in columns],
            "unit": [unit(column) for column in columns],
            "native_or_derived_temporal_support": [
                "monthly" if source_group(column) in {"SatPM target/history", "NOAA GSOD/ISD surface", "NOAA IGRA upper air", "derived time"}
                else "annual repeated within month"
                for column in columns
            ],
            "dtype_in_csv_read": [dtypes.get(column, "unknown") for column in columns],
            "missing_rows": [int(missing[column]) for column in columns],
            "missing_fraction": [float(missing[column] / rows) for column in columns],
        }
    )
    dictionary.to_csv(OUTPUT, index=False, encoding="utf-8-sig")
    print(f"wrote {len(dictionary)} variables for {rows:,} panel rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
