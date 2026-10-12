from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


START_YEAR = 2000
END_YEAR = 2024
ROOT = Path(__file__).resolve().parents[1]
MONTHLY_PM25_DIR = ROOT / "data/raw/pm25/satpm_v6gl03_coarse_monthly"
ANNUAL_GRID_PANEL = (
    ROOT / "data/processed/modelling_panel/gridcell_pm25_edgar_power_spatial_lagged_1998_2022.csv"
)
SURFACE_FUA_MONTH = ROOT / "data/interim/aviation/gsod_fua_month_2000_2025.csv.gz"
UPPER_AIR_FUA_MONTH = ROOT / "data/interim/aviation/igra_fua_month_2000_2025.csv.gz"
OUT_DIR = ROOT / "data/processed/modelling_panel"
META_DIR = ROOT / "data/metadata"
OUT_PATH = OUT_DIR / "gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
ANNUAL_AUX_PATH = OUT_DIR / "gridcell_annual_auxiliary_from_monthly_2000_2024.csv.gz"
META_PATH = META_DIR / "gridcell_monthly_pm25_multisource_2000_2024.json"

TARGET = "pm25_mean_ug_m3"
STATIC_COLUMNS = [
    "cell_id",
    "eFUA_ID",
    "fua_name",
    "country_iso",
    "country_name",
    "grid_lon",
    "grid_lat",
    "cell_source",
    "fua_area_km2",
    "urban_centre_area_km2",
    "fua_population_2015",
    "urban_centre_population_2015",
    "commuting_zone_population_2015",
    "population_density_2015_per_km2",
    "urban_centre_area_share",
    "urban_centre_population_share",
    "fua_centroid_lon",
    "fua_centroid_lat",
    "grid_delta_lon_from_fua_centroid",
    "grid_delta_lat_from_fua_centroid",
    "distance_to_coast_km",
    "is_coastal_50km",
    "is_coastal_100km",
]
ANNUAL_COVARIATE_COLUMNS = [
    "edgar_pm25_emissions_tonnes_gridcell",
    "edgar_pm25_emissions_positive_gridcell",
    "edgar_nox_emissions_tonnes_gridcell",
    "edgar_nox_emissions_positive_gridcell",
    "edgar_so2_emissions_tonnes_gridcell",
    "edgar_so2_emissions_positive_gridcell",
    "edgar_nh3_emissions_tonnes_gridcell",
    "edgar_nh3_emissions_positive_gridcell",
    "edgar_nmvoc_emissions_tonnes_gridcell",
    "edgar_nmvoc_emissions_positive_gridcell",
    "power_elevation_m",
    "power_prectotcorr_mm_day",
    "power_rh2m_percent",
    "power_t2m_c",
    "power_t2m_max_c",
    "power_t2m_min_c",
    "power_ws10m_m_s",
]


def monthly_key(path: Path) -> tuple[int, int]:
    match = re.search(r"\.(\d{4})(\d{2})-\d{6}\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse SatPM month from {path.name}")
    return int(match.group(1)), int(match.group(2))


def monthly_files() -> dict[tuple[int, int], Path]:
    files = {monthly_key(path): path for path in MONTHLY_PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc")}
    expected = {(year, month) for year in range(START_YEAR, END_YEAR + 1) for month in range(1, 13)}
    missing = sorted(expected - set(files))
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} SatPM months; first missing: {missing[:12]}")
    return files


def load_annual_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    usecols = list(dict.fromkeys(STATIC_COLUMNS + ["year", TARGET] + ANNUAL_COVARIATE_COLUMNS))
    annual = pd.read_csv(ANNUAL_GRID_PANEL, usecols=usecols)
    annual["eFUA_ID"] = annual["eFUA_ID"].astype(float)
    cells = annual.sort_values("year").drop_duplicates("cell_id")[STATIC_COLUMNS].copy()

    covariates = annual[["cell_id", "year"] + ANNUAL_COVARIATE_COLUMNS].copy()
    covariates["annual_covariate_reference_year"] = covariates["year"]
    covariates["annual_covariates_carried_forward"] = False
    reference_2022 = covariates[covariates["year"] == 2022].copy()
    carried = []
    for year in [2023, 2024]:
        clone = reference_2022.copy()
        clone["year"] = year
        clone["annual_covariates_carried_forward"] = True
        carried.append(clone)
    covariates = pd.concat([covariates, *carried], ignore_index=True)
    covariates = covariates[covariates["year"].between(START_YEAR, END_YEAR)]

    annual_target = annual[["cell_id", "year", TARGET]].rename(columns={TARGET: "pm25_official_annual_ug_m3"})
    return cells, covariates, annual_target


def sample_pm25(path: Path, cells: pd.DataFrame) -> np.ndarray:
    with xr.open_dataset(path) as dataset:
        array = dataset["PM25"]
        lons = xr.DataArray(cells["grid_lon"].to_numpy(dtype=float), dims="cell")
        lats = xr.DataArray(cells["grid_lat"].to_numpy(dtype=float), dims="cell")
        return array.sel(lon=lons, lat=lats, method="nearest").to_numpy().astype(float)


def build_panel() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    files = monthly_files()
    cells, covariates, annual_target = load_annual_inputs()
    surface = pd.read_csv(SURFACE_FUA_MONTH, compression="gzip")
    surface["eFUA_ID"] = surface["eFUA_ID"].astype(float)
    surface = surface.drop(columns=["eFUA_name", "Cntry_ISO"], errors="ignore")
    upper_air = pd.read_csv(UPPER_AIR_FUA_MONTH, compression="gzip")
    upper_air["eFUA_ID"] = upper_air["eFUA_ID"].astype(float)
    upper_air = upper_air.drop(columns=["eFUA_name", "Cntry_ISO"], errors="ignore")

    frames: list[pd.DataFrame] = []
    for sequence, ((year, month), path) in enumerate(sorted(files.items()), start=1):
        frame = cells.copy()
        frame["year"] = year
        frame["month"] = month
        frame["date"] = pd.Timestamp(year=year, month=month, day=1)
        frame[TARGET] = sample_pm25(path, frame)
        frame = frame.merge(covariates, on=["cell_id", "year"], how="left", validate="one_to_one")
        frame = frame.merge(surface, on=["eFUA_ID", "year", "month"], how="left", validate="many_to_one")
        frame = frame.merge(upper_air, on=["eFUA_ID", "year", "month"], how="left", validate="many_to_one")
        frames.append(frame)
        if sequence % 12 == 0:
            print(f"sampled SatPM through {year}-{month:02d}: {sequence}/300 files", flush=True)

    panel = pd.concat(frames, ignore_index=True)
    for column in ["surface_source_available", "aviation_icao_source_available", "igra_source_available"]:
        panel[column] = panel[column].fillna(False).astype(bool)
        panel[f"{column}_missing_flag"] = ~panel[column]
    for column in [
        "surface_station_count",
        "surface_eligible_station_count",
        "aviation_icao_station_count",
        "aviation_icao_eligible_station_count",
        "surface_valid_station_days",
        "igra_station_count",
        "igra_recommended_station_count",
        "igra_feature_count_max",
    ]:
        panel[column] = panel[column].fillna(0).astype(int)

    panel["month_sin"] = np.sin(2 * np.pi * panel["month"] / 12.0)
    panel["month_cos"] = np.cos(2 * np.pi * panel["month"] / 12.0)
    panel["time_index_month"] = (panel["year"] - START_YEAR) * 12 + panel["month"] - 1
    panel = panel.sort_values(["cell_id", "date"]).reset_index(drop=True)
    grouped = panel.groupby("cell_id", sort=False)[TARGET]
    for lag in [1, 3, 6, 12]:
        panel[f"pm25_lag{lag}_month"] = grouped.shift(lag)
    panel["pm25_roll3_mean_lag1_month"] = grouped.transform(
        lambda values: values.shift(1).rolling(3, min_periods=2).mean()
    )
    panel["pm25_roll12_mean_lag1_month"] = grouped.transform(
        lambda values: values.shift(1).rolling(12, min_periods=6).mean()
    )
    panel["pm25_change_lag1_month"] = grouped.diff().groupby(panel["cell_id"]).shift(1)

    annual_auxiliary = (
        panel.groupby(["cell_id", "eFUA_ID", "fua_name", "country_iso", "year"], as_index=False)[TARGET]
        .mean()
        .rename(columns={TARGET: "pm25_monthly_derived_annual_ug_m3"})
        .merge(annual_target, on=["cell_id", "year"], how="left", validate="one_to_one")
    )
    annual_auxiliary["monthly_minus_official_annual_ug_m3"] = (
        annual_auxiliary["pm25_monthly_derived_annual_ug_m3"]
        - annual_auxiliary["pm25_official_annual_ug_m3"]
    )
    comparable = annual_auxiliary.dropna(subset=["pm25_official_annual_ug_m3"])
    difference = comparable["monthly_minus_official_annual_ug_m3"]
    consistency = {
        "comparable_rows": int(len(comparable)),
        "mean_difference_ug_m3": float(difference.mean()),
        "mae_ug_m3": float(difference.abs().mean()),
        "rmse_ug_m3": float(np.sqrt(np.mean(np.square(difference)))),
        "correlation": float(
            comparable[["pm25_monthly_derived_annual_ug_m3", "pm25_official_annual_ug_m3"]].corr().iloc[0, 1]
        ),
    }
    return panel, annual_auxiliary, consistency


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    panel, annual_auxiliary, consistency = build_panel()
    panel.to_csv(OUT_PATH, index=False, compression="gzip", encoding="utf-8", float_format="%.7g")
    annual_auxiliary.to_csv(
        ANNUAL_AUX_PATH, index=False, compression="gzip", encoding="utf-8", float_format="%.7g"
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study_period": [START_YEAR, END_YEAR],
        "temporal_unit": "month",
        "rows": int(len(panel)),
        "cell_count": int(panel["cell_id"].nunique()),
        "fua_count": int(panel["eFUA_ID"].nunique()),
        "month_count": int(panel[["year", "month"]].drop_duplicates().shape[0]),
        "country_rows": panel["country_iso"].value_counts().to_dict(),
        "surface_source_rows": int(panel["surface_source_available"].sum()),
        "aviation_icao_source_rows": int(panel["aviation_icao_source_available"].sum()),
        "igra_source_rows": int(panel["igra_source_available"].sum()),
        "annual_covariate_policy": (
            "Observed annual covariates are repeated within year through 2022; 2022 is explicitly carried forward "
            "to 2023-2024 with a provenance flag."
        ),
        "annual_auxiliary_consistency": consistency,
        "outputs": [str(OUT_PATH.relative_to(ROOT)), str(ANNUAL_AUX_PATH.relative_to(ROOT))],
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
