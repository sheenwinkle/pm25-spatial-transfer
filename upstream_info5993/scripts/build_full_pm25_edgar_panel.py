from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
PM25_DIR = Path("data/raw/pm25/satpm_v6gl03_coarse_annual")
EDGAR_DIR = Path("data/interim/emissions/edgar_v81_pm25_totals_2012_2022")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

YEARS = range(2012, 2023)
COUNTRIES = ["CHN", "AUS"]


def satpm_year_from_name(path: Path) -> int:
    match = re.search(r"\.(\d{4})01-\d{4}12\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse SatPM year from {path.name}")
    return int(match.group(1))


def edgar_year_from_name(path: Path) -> int:
    match = re.search(r"_(\d{4})_TOTALS_emi\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse EDGAR year from {path.name}")
    return int(match.group(1))


def load_fuas() -> gpd.GeoDataFrame:
    fua_path = f"zip://{FUA_ZIP}!{FUA_LAYER}"
    fua = gpd.read_file(fua_path)
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    fua = fua.to_crs("EPSG:4326")
    fua["eFUA_ID"] = fua["eFUA_ID"].astype(float)
    return fua


def points_from_grid(lats, lons, values, value_name: str, positive_only: bool) -> gpd.GeoDataFrame:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_values = np.asarray(values).ravel()
    mask = np.isfinite(flat_values)
    if positive_only:
        mask = mask & (flat_values > 0)
    else:
        mask = mask & (flat_values >= 0)

    frame = pd.DataFrame(
        {
            "lon": lon_grid.ravel()[mask],
            "lat": lat_grid.ravel()[mask],
            value_name: flat_values[mask],
        }
    )
    return gpd.GeoDataFrame(
        frame,
        geometry=gpd.points_from_xy(frame["lon"], frame["lat"]),
        crs="EPSG:4326",
    )


def subset_array_to_fuas(arr: xr.DataArray, fuas: gpd.GeoDataFrame) -> xr.DataArray:
    minx, miny, maxx, maxy = fuas.total_bounds
    return arr.sel(
        lon=slice(float(minx) - 0.2, float(maxx) + 0.2),
        lat=slice(float(miny) - 0.2, float(maxy) + 0.2),
    )


def aggregate_pm25(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    records = []
    pm25_files = {
        satpm_year_from_name(path): path
        for path in PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc")
    }

    for year in YEARS:
        ds = xr.open_dataset(pm25_files[year])
        arr = ds["PM25"]
        for country in COUNTRIES:
            country_fuas = fuas[fuas["Cntry_ISO"] == country][["eFUA_ID", "geometry"]]
            sub = subset_array_to_fuas(arr, country_fuas)
            pts = points_from_grid(
                sub["lat"].values,
                sub["lon"].values,
                sub.values,
                "pm25_mean_ug_m3",
                positive_only=False,
            )
            joined = gpd.sjoin(
                pts,
                country_fuas,
                how="inner",
                predicate="within",
            )
            grouped = (
                joined.groupby("eFUA_ID")["pm25_mean_ug_m3"]
                .agg(pm25_mean_ug_m3="mean", pm25_grid_cell_count="count")
                .reset_index()
            )
            grouped["year"] = year
            records.append(grouped)

    return pd.concat(records, ignore_index=True)


def aggregate_edgar_pm25(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    records = []
    edgar_files = {
        edgar_year_from_name(path): path
        for path in EDGAR_DIR.glob("v8.1_FT2022_AP_PM2.5_*_TOTALS_emi.nc")
    }

    for year in YEARS:
        ds = xr.open_dataset(edgar_files[year])
        arr = ds["emissions"]
        for country in COUNTRIES:
            country_fuas = fuas[fuas["Cntry_ISO"] == country][["eFUA_ID", "geometry"]]
            sub = subset_array_to_fuas(arr, country_fuas)
            pts = points_from_grid(
                sub["lat"].values,
                sub["lon"].values,
                sub.values,
                "edgar_pm25_emissions_tonnes",
                positive_only=True,
            )
            joined = gpd.sjoin(
                pts,
                country_fuas,
                how="inner",
                predicate="within",
            )
            grouped = (
                joined.groupby("eFUA_ID")["edgar_pm25_emissions_tonnes"]
                .agg(
                    edgar_pm25_emissions_tonnes="sum",
                    edgar_positive_grid_cell_mean_tonnes="mean",
                    edgar_positive_grid_cell_count="count",
                )
                .reset_index()
            )
            grouped["year"] = year
            records.append(grouped)

    return pd.concat(records, ignore_index=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    fuas = load_fuas()
    attrs = fuas.drop(columns="geometry").copy()
    attrs_path = Path("data/interim/ghs_fua_china_australia_attributes.csv")
    attrs_path.parent.mkdir(parents=True, exist_ok=True)
    attrs.to_csv(attrs_path, index=False, encoding="utf-8-sig")

    base = attrs[
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
        ]
    ].rename(
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
    base["eFUA_ID"] = base["eFUA_ID"].astype(float)
    base["population_density_2015_per_km2"] = (
        base["fua_population_2015"] / base["fua_area_km2"]
    )

    panel_base = pd.MultiIndex.from_product(
        [base["eFUA_ID"], YEARS], names=["eFUA_ID", "year"]
    ).to_frame(index=False)
    panel = panel_base.merge(base, on="eFUA_ID", how="left")

    pm25 = aggregate_pm25(fuas)
    edgar = aggregate_edgar_pm25(fuas)

    panel = panel.merge(pm25, on=["eFUA_ID", "year"], how="left")
    panel = panel.merge(edgar, on=["eFUA_ID", "year"], how="left")
    panel["edgar_pm25_emissions_tonnes"] = panel["edgar_pm25_emissions_tonnes"].fillna(0)
    panel["edgar_positive_grid_cell_count"] = panel[
        "edgar_positive_grid_cell_count"
    ].fillna(0)
    panel["edgar_pm25_emissions_tonnes_per_km2"] = (
        panel["edgar_pm25_emissions_tonnes"] / panel["fua_area_km2"]
    )

    panel = panel.sort_values(["country_iso", "fua_population_2015", "eFUA_ID", "year"])
    out_path = OUT_DIR / "full_china_australia_pm25_edgar_pm25_ghs_fua_2012_2022.csv"
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")

    complete_ids = (
        panel.groupby("eFUA_ID")["pm25_mean_ug_m3"]
        .apply(lambda series: series.notna().sum() == len(YEARS))
    )
    complete_ids = complete_ids[complete_ids].index
    complete_panel = panel[panel["eFUA_ID"].isin(complete_ids)].copy()
    complete_path = (
        OUT_DIR
        / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022.csv"
    )
    complete_panel.to_csv(complete_path, index=False, encoding="utf-8-sig")

    summary = (
        panel.groupby("country_iso")
        .agg(
            fua_count=("eFUA_ID", "nunique"),
            row_count=("eFUA_ID", "count"),
            pm25_nonmissing=("pm25_mean_ug_m3", "count"),
            pm25_mean=("pm25_mean_ug_m3", "mean"),
            emissions_mean_tonnes=("edgar_pm25_emissions_tonnes", "mean"),
            emissions_density_mean=("edgar_pm25_emissions_tonnes_per_km2", "mean"),
        )
        .reset_index()
    )
    summary_path = OUT_DIR / "full_china_australia_pm25_edgar_pm25_ghs_fua_2012_2022_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_source": str(FUA_ZIP),
        "pm25_source_dir": str(PM25_DIR),
        "edgar_source_dir": str(EDGAR_DIR),
        "countries": COUNTRIES,
        "years": list(YEARS),
        "method": "Full China/Australia GHS-FUA panel. SatPM PM2.5 is mean concentration over 0.1 degree grid-cell centres within each FUA. EDGAR PM2.5 is summed positive grid-cell emissions in tonnes within each FUA and divided by GHS-FUA area for density.",
        "output_panel": str(out_path),
        "output_complete_pm25_panel": str(complete_path),
        "output_summary": str(summary_path),
        "rows": int(len(panel)),
        "fua_count": int(panel["eFUA_ID"].nunique()),
        "complete_pm25_rows": int(len(complete_panel)),
        "complete_pm25_fua_count": int(complete_panel["eFUA_ID"].nunique()),
    }
    (META_DIR / "full_china_australia_pm25_edgar_pm25_ghs_fua_2012_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string(index=False))
    print(f"\nOutput panel: {out_path}")
    print(f"Complete PM2.5 panel: {complete_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
