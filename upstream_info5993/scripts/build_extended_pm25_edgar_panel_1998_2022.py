from __future__ import annotations

import json
import re
import sys
import zipfile
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
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

YEARS = range(1998, 2023)
COUNTRIES = ["CHN", "AUS"]
POLLUTANTS = {
    "PM2.5": "pm25",
    "NOx": "nox",
    "SO2": "so2",
    "NH3": "nh3",
    "NMVOC": "nmvoc",
}


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
    fua = gpd.read_file(f"zip://{FUA_ZIP}!{FUA_LAYER}")
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    fua["eFUA_ID"] = fua["eFUA_ID"].astype(float)
    return fua.to_crs("EPSG:4326")


def points_from_grid(lats, lons, values, value_name: str, positive_only: bool) -> gpd.GeoDataFrame:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_values = np.asarray(values).ravel()
    mask = np.isfinite(flat_values)
    mask = mask & (flat_values > 0 if positive_only else flat_values >= 0)
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


def subset_array(arr: xr.DataArray, fuas: gpd.GeoDataFrame) -> xr.DataArray:
    minx, miny, maxx, maxy = fuas.total_bounds
    return arr.sel(
        lon=slice(float(minx) - 0.2, float(maxx) + 0.2),
        lat=slice(float(miny) - 0.2, float(maxy) + 0.2),
    )


def base_panel(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    attrs = fuas.drop(columns="geometry").copy()
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
    base["population_density_2015_per_km2"] = (
        base["fua_population_2015"] / base["fua_area_km2"]
    )
    frame = pd.MultiIndex.from_product(
        [base["eFUA_ID"], YEARS], names=["eFUA_ID", "year"]
    ).to_frame(index=False)
    return frame.merge(base, on="eFUA_ID", how="left")


def aggregate_pm25(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    files = {
        satpm_year_from_name(path): path
        for path in PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc")
    }
    missing_files = [year for year in YEARS if year not in files]
    if missing_files:
        raise FileNotFoundError(f"Missing SatPM years: {missing_files}")

    centroids = gpd.GeoDataFrame(
        fuas[["eFUA_ID", "Cntry_ISO"]].copy(),
        geometry=fuas.to_crs("EPSG:3857").geometry.centroid,
        crs="EPSG:3857",
    ).to_crs("EPSG:4326")
    records = []

    for year in YEARS:
        ds = xr.open_dataset(files[year])
        arr = ds["PM25"]
        for country in COUNTRIES:
            country_fuas = fuas[fuas["Cntry_ISO"] == country][["eFUA_ID", "geometry"]]
            sub = subset_array(arr, country_fuas)
            pts = points_from_grid(
                sub["lat"].values,
                sub["lon"].values,
                sub.values,
                "pm25_mean_ug_m3",
                positive_only=False,
            )
            joined = gpd.sjoin(pts, country_fuas, how="inner", predicate="within")
            grouped = (
                joined.groupby("eFUA_ID")["pm25_mean_ug_m3"]
                .agg(pm25_mean_ug_m3="mean", pm25_grid_cell_count="count")
                .reset_index()
            )
            grouped["year"] = year
            grouped["pm25_aggregation_method"] = "polygon_grid_centres"

            missing_ids = set(country_fuas["eFUA_ID"]) - set(grouped["eFUA_ID"])
            if missing_ids:
                fallback_rows = []
                missing_centroids = centroids[centroids["eFUA_ID"].isin(missing_ids)]
                for _, row in missing_centroids.iterrows():
                    value = arr.sel(
                        lon=float(row.geometry.x),
                        lat=float(row.geometry.y),
                        method="nearest",
                    )
                    fallback_rows.append(
                        {
                            "eFUA_ID": row["eFUA_ID"],
                            "pm25_mean_ug_m3": float(value.values),
                            "pm25_grid_cell_count": 0,
                            "year": year,
                            "pm25_aggregation_method": "nearest_centroid_grid_cell",
                        }
                    )
                grouped = pd.concat([grouped, pd.DataFrame(fallback_rows)], ignore_index=True)
            records.append(grouped)

    return pd.concat(records, ignore_index=True)


def edgar_zip_path(slug: str) -> Path:
    return Path(f"data/raw/emissions/edgar_v81_{slug}_totals/TOTALS_emi_nc.zip")


def extract_edgar_years(pollutant: str, slug: str) -> list[Path]:
    zip_path = edgar_zip_path(slug)
    extract_dir = Path(f"data/interim/emissions/edgar_v81_{slug}_totals_1998_2022")
    extract_dir.mkdir(parents=True, exist_ok=True)
    paths = []
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
                    dst.write(src.read())
            paths.append(out_path)
    return paths


def aggregate_edgar_pollutant(fuas: gpd.GeoDataFrame, pollutant: str, slug: str) -> pd.DataFrame:
    value_col = f"edgar_{slug}_emissions_tonnes"
    count_col = f"edgar_{slug}_positive_grid_cell_count"
    mean_col = f"edgar_{slug}_positive_grid_cell_mean_tonnes"
    density_col = f"edgar_{slug}_emissions_tonnes_per_km2"
    nc_paths = extract_edgar_years(pollutant, slug)
    records = []

    for nc_path in sorted(nc_paths, key=nc_year_from_name):
        year = nc_year_from_name(nc_path)
        ds = xr.open_dataset(nc_path)
        arr = ds["emissions"]
        for country in COUNTRIES:
            country_fuas = fuas[fuas["Cntry_ISO"] == country][["eFUA_ID", "FUA_area", "geometry"]]
            sub = subset_array(arr, country_fuas)
            pts = points_from_grid(
                sub["lat"].values,
                sub["lon"].values,
                sub.values,
                value_col,
                positive_only=True,
            )
            joined = gpd.sjoin(
                pts,
                country_fuas[["eFUA_ID", "geometry"]],
                how="inner",
                predicate="within",
            )
            grouped = (
                joined.groupby("eFUA_ID")[value_col]
                .agg(**{value_col: "sum", mean_col: "mean", count_col: "count"})
                .reset_index()
            )
            grouped["year"] = year
            records.append(grouped)

    panel = pd.concat(records, ignore_index=True)
    complete_index = pd.MultiIndex.from_product(
        [fuas["eFUA_ID"].unique(), YEARS], names=["eFUA_ID", "year"]
    ).to_frame(index=False)
    area = fuas[["eFUA_ID", "FUA_area"]].rename(columns={"FUA_area": "fua_area_km2"})
    panel = complete_index.merge(panel, on=["eFUA_ID", "year"], how="left")
    panel[value_col] = panel[value_col].fillna(0)
    panel[count_col] = panel[count_col].fillna(0)
    panel = panel.merge(area, on="eFUA_ID", how="left")
    panel[density_col] = panel[value_col] / panel["fua_area_km2"]
    return panel.drop(columns=["fua_area_km2"])


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    fuas = load_fuas()
    panel = base_panel(fuas)
    panel = panel.merge(aggregate_pm25(fuas), on=["eFUA_ID", "year"], how="left")

    for pollutant, slug in POLLUTANTS.items():
        print(f"Aggregating EDGAR {pollutant}", flush=True)
        edgar = aggregate_edgar_pollutant(fuas, pollutant, slug)
        panel = panel.merge(edgar, on=["eFUA_ID", "year"], how="left")

    out_path = OUT_DIR / "extended_all_fua_pm25_edgar_ghs_fua_1998_2022.csv"
    panel = panel.sort_values(["country_iso", "eFUA_ID", "year"])
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = panel.groupby("country_iso").agg(
        fua_count=("eFUA_ID", "nunique"),
        rows=("eFUA_ID", "count"),
        pm25_nonmissing=("pm25_mean_ug_m3", "count"),
        nearest_pm25_rows=(
            "pm25_aggregation_method",
            lambda s: int((s == "nearest_centroid_grid_cell").sum()),
        ),
        pm25_mean=("pm25_mean_ug_m3", "mean"),
    )
    summary_path = OUT_DIR / "extended_all_fua_pm25_edgar_ghs_fua_1998_2022_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "years": list(YEARS),
        "countries": COUNTRIES,
        "fua_count": int(panel["eFUA_ID"].nunique()),
        "rows": int(len(panel)),
        "method": "Extended all-FUA panel. PM2.5 uses polygon grid-centre means when available and nearest centroid grid-cell fallback for small FUAs. EDGAR pollutant emissions are summed over positive 0.1 degree grid-cell centres inside each FUA.",
        "output_panel": str(out_path),
        "summary_output": str(summary_path),
    }
    (META_DIR / "extended_all_fua_pm25_edgar_ghs_fua_1998_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string())
    print(f"\nOutput panel: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
