from __future__ import annotations

import json
import re
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


RANDOM_STATE = 5993
TARGET = "pm25_mean_ug_m3"
YEARS = range(1998, 2023)
COUNTRIES = ["CHN", "AUS"]
POLLUTANTS = {
    "PM2.5": "pm25",
    "NOx": "nox",
    "SO2": "so2",
    "NH3": "nh3",
    "NMVOC": "nmvoc",
}

FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
PM25_DIR = Path("data/raw/pm25/satpm_v6gl03_coarse_annual")
FUA_PANEL = Path("data/processed/modelling_panel/extended_all_fua_pm25_edgar_power_spatial_ghs_fua_1998_2022.csv")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")


def satpm_year_from_name(path: Path) -> int:
    match = re.search(r"\.(\d{4})01-\d{4}12\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse SatPM year from {path.name}")
    return int(match.group(1))


def nc_year_from_name(path: Path | str) -> int:
    match = re.search(r"_(\d{4})_TOTALS_emi\.nc$", str(path))
    if not match:
        raise ValueError(f"Cannot parse EDGAR year from {path}")
    return int(match.group(1))


def load_fuas() -> gpd.GeoDataFrame:
    fuas = gpd.read_file(f"zip://{FUA_ZIP}!{FUA_LAYER}")
    fuas = fuas[fuas["Cntry_ISO"].isin(COUNTRIES)].copy()
    fuas["eFUA_ID"] = fuas["eFUA_ID"].astype(float)
    return fuas.to_crs("EPSG:4326")


def subset_array(arr: xr.DataArray, fuas: gpd.GeoDataFrame) -> xr.DataArray:
    minx, miny, maxx, maxy = fuas.total_bounds
    return arr.sel(
        lon=slice(float(minx) - 0.2, float(maxx) + 0.2),
        lat=slice(float(miny) - 0.2, float(maxy) + 0.2),
    )


def build_cell_index(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    sample_file = sorted(PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc"))[0]
    arr = xr.open_dataset(sample_file)["PM25"]
    frames = []

    fua_attrs = fuas[
        [
            "eFUA_ID",
            "eFUA_name",
            "Cntry_ISO",
            "Cntry_name",
            "FUA_area",
            "UC_area",
            "FUA_p_2015",
            "UC_p_2015",
            "Com_p_2015",
            "geometry",
        ]
    ].copy()

    centroids = gpd.GeoDataFrame(
        fuas[["eFUA_ID", "Cntry_ISO"]].copy(),
        geometry=fuas.to_crs("EPSG:3857").geometry.centroid,
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")

    for country in COUNTRIES:
        country_fuas = fua_attrs[fua_attrs["Cntry_ISO"] == country].copy()
        sub = subset_array(arr, country_fuas)
        lon_grid, lat_grid = np.meshgrid(sub["lon"].values, sub["lat"].values)
        pts = pd.DataFrame({"grid_lon": lon_grid.ravel(), "grid_lat": lat_grid.ravel()})
        gpts = gpd.GeoDataFrame(
            pts,
            geometry=gpd.points_from_xy(pts["grid_lon"], pts["grid_lat"]),
            crs="EPSG:4326",
        )
        joined = gpd.sjoin(
            gpts,
            country_fuas[["eFUA_ID", "geometry"]],
            how="inner",
            predicate="within",
        ).drop(columns=["index_right"])
        joined["cell_source"] = "polygon_grid_centres"

        missing = sorted(set(country_fuas["eFUA_ID"]) - set(joined["eFUA_ID"]))
        if missing:
            fallback_rows = []
            for _, row in centroids[centroids["eFUA_ID"].isin(missing)].iterrows():
                lon_val = float(arr.sel(lon=float(row.geometry.x), method="nearest")["lon"].values)
                lat_val = float(arr.sel(lat=float(row.geometry.y), method="nearest")["lat"].values)
                fallback_rows.append(
                    {
                        "grid_lon": lon_val,
                        "grid_lat": lat_val,
                        "geometry": row.geometry,
                        "eFUA_ID": float(row["eFUA_ID"]),
                        "cell_source": "nearest_fua_centroid_grid_cell",
                    }
                )
            joined = pd.concat([joined, gpd.GeoDataFrame(fallback_rows, crs="EPSG:4326")], ignore_index=True)
        frames.append(joined)

    cells = pd.concat(frames, ignore_index=True)
    cells = cells.merge(fua_attrs.drop(columns="geometry"), on="eFUA_ID", how="left")
    cells = cells.rename(
        columns={
            "eFUA_name": "fua_name",
            "Cntry_ISO": "country_iso",
            "Cntry_name": "country_name",
            "FUA_area": "fua_area_km2",
            "UC_area": "urban_centre_area_km2",
            "FUA_p_2015": "fua_population_2015",
            "UC_p_2015": "urban_centre_population_2015",
            "Com_p_2015": "commuting_zone_population_2015",
        }
    )
    cells["grid_lon"] = cells["grid_lon"].astype(float).round(4)
    cells["grid_lat"] = cells["grid_lat"].astype(float).round(4)
    cells["cell_id"] = (
        cells["eFUA_ID"].astype(int).astype(str)
        + "_"
        + cells["grid_lon"].map(lambda x: f"{x:.2f}")
        + "_"
        + cells["grid_lat"].map(lambda x: f"{x:.2f}")
    )
    cells["population_density_2015_per_km2"] = cells["fua_population_2015"] / cells["fua_area_km2"]
    cells["urban_centre_area_share"] = cells["urban_centre_area_km2"] / cells["fua_area_km2"]
    cells["urban_centre_population_share"] = cells["urban_centre_population_2015"] / cells["fua_population_2015"]

    cent = fuas.to_crs("EPSG:3857").copy()
    cent["centroid_geom"] = cent.geometry.centroid
    centroids_wgs = gpd.GeoDataFrame(
        cent[["eFUA_ID"]],
        geometry=cent["centroid_geom"],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    centroids_wgs["fua_centroid_lon"] = centroids_wgs.geometry.x
    centroids_wgs["fua_centroid_lat"] = centroids_wgs.geometry.y
    cells = cells.merge(
        centroids_wgs.drop(columns="geometry"),
        on="eFUA_ID",
        how="left",
    )
    cells["grid_delta_lon_from_fua_centroid"] = cells["grid_lon"] - cells["fua_centroid_lon"]
    cells["grid_delta_lat_from_fua_centroid"] = cells["grid_lat"] - cells["fua_centroid_lat"]
    return pd.DataFrame(cells.drop(columns="geometry", errors="ignore"))


def sample_year_array(arr: xr.DataArray, cells: pd.DataFrame) -> np.ndarray:
    lons = xr.DataArray(cells["grid_lon"].to_numpy(dtype=float), dims="cell")
    lats = xr.DataArray(cells["grid_lat"].to_numpy(dtype=float), dims="cell")
    return arr.sel(lon=lons, lat=lats, method="nearest").to_numpy()


def pm25_files() -> dict[int, Path]:
    files = {satpm_year_from_name(path): path for path in PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc")}
    missing = [year for year in YEARS if year not in files]
    if missing:
        raise FileNotFoundError(f"Missing SatPM years: {missing}")
    return files


def edgar_zip_path(slug: str) -> Path:
    return Path(f"data/raw/emissions/edgar_v81_{slug}_totals/TOTALS_emi_nc.zip")


def extract_edgar_years(pollutant: str, slug: str) -> dict[int, Path]:
    zip_path = edgar_zip_path(slug)
    extract_dir = Path(f"data/interim/emissions/edgar_v81_{slug}_totals_1998_2022")
    extract_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = {
            nc_year_from_name(name): name
            for name in zf.namelist()
            if name.endswith("_TOTALS_emi.nc") and f"_{pollutant}_" in name
        }
        missing = [year for year in YEARS if year not in names]
        if missing:
            raise FileNotFoundError(f"Missing EDGAR {pollutant} years: {missing}")
        for year in YEARS:
            name = names[year]
            out_path = extract_dir / Path(name).name
            info = zf.getinfo(name)
            if not out_path.exists() or out_path.stat().st_size != info.file_size:
                with zf.open(name) as src, out_path.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            paths[year] = out_path
    return paths


def build_grid_panel() -> pd.DataFrame:
    fuas = load_fuas()
    cells = build_cell_index(fuas)
    static_cols = [
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
    ]
    cells = cells[static_cols].copy()

    pm_files = pm25_files()
    edgar_files = {slug: extract_edgar_years(pollutant, slug) for pollutant, slug in POLLUTANTS.items()}

    records = []
    for year in YEARS:
        frame = cells.copy()
        frame["year"] = year
        pm_arr = xr.open_dataset(pm_files[year])["PM25"]
        frame[TARGET] = sample_year_array(pm_arr, frame).astype(float)
        for slug, by_year in edgar_files.items():
            arr = xr.open_dataset(by_year[year])["emissions"]
            values = sample_year_array(arr, frame).astype(float)
            values = np.where(np.isfinite(values), values, 0.0)
            frame[f"edgar_{slug}_emissions_tonnes_gridcell"] = values
            frame[f"edgar_{slug}_emissions_positive_gridcell"] = (values > 0).astype(int)
        records.append(frame)
        print(f"built grid year {year}: {len(frame):,} rows")

    panel = pd.concat(records, ignore_index=True)

    fua_panel = pd.read_csv(FUA_PANEL)
    join_cols = [
        "eFUA_ID",
        "year",
        "power_elevation_m",
        "power_prectotcorr_mm_day",
        "power_rh2m_percent",
        "power_t2m_c",
        "power_t2m_max_c",
        "power_t2m_min_c",
        "power_ws10m_m_s",
        "distance_to_coast_km",
        "is_coastal_50km",
        "is_coastal_100km",
    ]
    panel = panel.merge(fua_panel[join_cols], on=["eFUA_ID", "year"], how="left")

    panel = panel.sort_values(["cell_id", "year"]).copy()
    group = panel.groupby("cell_id", sort=False)
    panel["pm25_lag1"] = group[TARGET].shift(1)
    panel["pm25_lag2"] = group[TARGET].shift(2)
    panel["pm25_lag3"] = group[TARGET].shift(3)
    panel["pm25_change_lag1"] = group[TARGET].diff(1).shift(1)
    panel["pm25_change_lag2"] = group[TARGET].diff(1).shift(2)
    panel["pm25_roll3_mean_lag1"] = group[TARGET].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    panel["pm25_roll5_mean_lag1"] = group[TARGET].transform(lambda s: s.shift(1).rolling(5, min_periods=2).mean())
    for slug in POLLUTANTS.values():
        col = f"edgar_{slug}_emissions_tonnes_gridcell"
        panel[f"{col}_lag1"] = group[col].shift(1)
        panel[f"{col}_change_lag1"] = group[col].diff(1).shift(1)

    return panel


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    panel = build_grid_panel()
    out_path = OUT_DIR / "gridcell_pm25_edgar_power_spatial_lagged_1998_2022.csv"
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")
    usable = panel.dropna(subset=["pm25_lag1", "pm25_lag2", "pm25_lag3"]).copy()
    usable_path = OUT_DIR / "gridcell_pm25_edgar_power_spatial_lagged_model_ready_2001_2022.csv"
    usable.to_csv(usable_path, index=False, encoding="utf-8-sig")
    summary = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "SatPM/ACAG V6.GL.03 0.1 degree PM2.5 + EDGAR v8.1 air-pollutant grids + FUA-level NASA POWER/spatial features",
        "full_output": str(out_path),
        "model_ready_output": str(usable_path),
        "full_rows": int(len(panel)),
        "model_ready_rows": int(len(usable)),
        "cell_count": int(panel["cell_id"].nunique()),
        "fua_count": int(panel["eFUA_ID"].nunique()),
        "country_rows_full": panel["country_iso"].value_counts().to_dict(),
        "country_cells": panel.drop_duplicates("cell_id")["country_iso"].value_counts().to_dict(),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
    }
    (META_DIR / "gridcell_pm25_edgar_power_spatial_lagged_1998_2022.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
