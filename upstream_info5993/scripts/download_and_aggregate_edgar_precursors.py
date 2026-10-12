from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import xarray as xr


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

BASE_URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/EDGAR/"
    "datasets/v81_FT2022_AP_new/{pollutant}/TOTALS/TOTALS_emi_nc.zip"
)
POLLUTANTS = {
    "NOx": "nox",
    "SO2": "so2",
    "NH3": "nh3",
    "NMVOC": "nmvoc",
}
YEARS = range(2012, 2023)
COUNTRIES = ["CHN", "AUS"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_zip(pollutant: str, slug: str) -> Path:
    raw_dir = Path(f"data/raw/emissions/edgar_v81_{slug}_totals")
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / "TOTALS_emi_nc.zip"
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    url = BASE_URL.format(pollutant=pollutant)
    part_path = out_path.with_suffix(".zip.part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part_path.replace(out_path)
    return out_path


def nc_year_from_name(path: Path | str) -> int:
    match = re.search(r"_(\d{4})_TOTALS_emi\.nc$", str(path))
    if not match:
        raise ValueError(f"Cannot parse EDGAR year from {path}")
    return int(match.group(1))


def extract_years(zip_path: Path, pollutant: str, slug: str) -> list[Path]:
    extract_dir = Path(f"data/interim/emissions/edgar_v81_{slug}_totals_2012_2022")
    extract_dir.mkdir(parents=True, exist_ok=True)
    extracted = []
    with zipfile.ZipFile(zip_path) as zf:
        names = {
            nc_year_from_name(name): name
            for name in zf.namelist()
            if name.endswith("_TOTALS_emi.nc") and f"_{pollutant}_" in name
        }
        missing = [year for year in YEARS if year not in names]
        if missing:
            raise FileNotFoundError(f"Missing {pollutant} years in {zip_path}: {missing}")
        for year in YEARS:
            name = names[year]
            out_path = extract_dir / Path(name).name
            info = zf.getinfo(name)
            if not out_path.exists() or out_path.stat().st_size != info.file_size:
                with zf.open(name) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
            extracted.append(out_path)
    return extracted


def load_fuas() -> gpd.GeoDataFrame:
    fua_path = f"zip://{FUA_ZIP}!{FUA_LAYER}"
    fua = gpd.read_file(fua_path)
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    fua = fua.to_crs("EPSG:4326")
    fua["eFUA_ID"] = fua["eFUA_ID"].astype(float)
    return fua


def subset_array_to_fuas(arr: xr.DataArray, fuas: gpd.GeoDataFrame) -> xr.DataArray:
    minx, miny, maxx, maxy = fuas.total_bounds
    return arr.sel(
        lon=slice(float(minx) - 0.2, float(maxx) + 0.2),
        lat=slice(float(miny) - 0.2, float(maxy) + 0.2),
    )


def points_from_grid(lats, lons, values, value_name: str) -> gpd.GeoDataFrame:
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    flat_values = np.asarray(values).ravel()
    mask = np.isfinite(flat_values) & (flat_values > 0)
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


def aggregate_pollutant(fuas: gpd.GeoDataFrame, nc_paths: list[Path], slug: str) -> pd.DataFrame:
    records = []
    value_col = f"edgar_{slug}_emissions_tonnes"
    mean_col = f"edgar_{slug}_positive_grid_cell_mean_tonnes"
    count_col = f"edgar_{slug}_positive_grid_cell_count"

    for nc_path in sorted(nc_paths, key=nc_year_from_name):
        year = nc_year_from_name(nc_path)
        ds = xr.open_dataset(nc_path)
        arr = ds["emissions"]
        for country in COUNTRIES:
            country_fuas = fuas[fuas["Cntry_ISO"] == country][["eFUA_ID", "FUA_area", "geometry"]]
            sub = subset_array_to_fuas(arr, country_fuas)
            pts = points_from_grid(sub["lat"].values, sub["lon"].values, sub.values, value_col)
            if pts.empty:
                grouped = pd.DataFrame(columns=["eFUA_ID", value_col, mean_col, count_col])
            else:
                joined = gpd.sjoin(pts, country_fuas[["eFUA_ID", "geometry"]], how="inner", predicate="within")
                grouped = (
                    joined.groupby("eFUA_ID")[value_col]
                    .agg(**{value_col: "sum", mean_col: "mean", count_col: "count"})
                    .reset_index()
                )
            grouped["year"] = year
            records.append(grouped)

    panel = pd.concat(records, ignore_index=True)
    base = pd.MultiIndex.from_product(
        [fuas["eFUA_ID"].unique(), YEARS], names=["eFUA_ID", "year"]
    ).to_frame(index=False)
    area = fuas[["eFUA_ID", "FUA_area"]].drop_duplicates().rename(columns={"FUA_area": "fua_area_km2"})
    panel = base.merge(panel, on=["eFUA_ID", "year"], how="left").merge(area, on="eFUA_ID", how="left")
    panel[value_col] = panel[value_col].fillna(0)
    panel[count_col] = panel[count_col].fillna(0)
    panel[f"edgar_{slug}_emissions_tonnes_per_km2"] = panel[value_col] / panel["fua_area_km2"]
    panel = panel.drop(columns=["fua_area_km2"])
    return panel


def merge_precursors(precursors: pd.DataFrame) -> dict[str, str]:
    outputs = {}
    targets = {
        "full": OUT_DIR / "full_china_australia_pm25_edgar_power_ghs_fua_2012_2022.csv",
        "complete": OUT_DIR
        / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power.csv",
    }
    for label, path in targets.items():
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        merged = panel.merge(precursors, on=["eFUA_ID", "year"], how="left")
        if label == "full":
            out_path = OUT_DIR / "full_china_australia_pm25_edgar_power_precursors_ghs_fua_2012_2022.csv"
        else:
            out_path = (
                OUT_DIR
                / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power_precursors.csv"
            )
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[label] = str(out_path)
    return outputs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    fuas = load_fuas()

    panels = []
    source_records = []
    for pollutant, slug in POLLUTANTS.items():
        print(f"Processing EDGAR {pollutant}")
        zip_path = download_zip(pollutant, slug)
        digest = sha256_file(zip_path)
        nc_paths = extract_years(zip_path, pollutant, slug)
        panel = aggregate_pollutant(fuas, nc_paths, slug)
        panels.append(panel)
        source_records.append(
            {
                "pollutant": pollutant,
                "slug": slug,
                "url": BASE_URL.format(pollutant=pollutant),
                "zip_path": str(zip_path),
                "zip_size_bytes": zip_path.stat().st_size,
                "zip_sha256": digest,
                "extracted_files": [str(path) for path in nc_paths],
            }
        )

    precursors = panels[0]
    for panel in panels[1:]:
        precursors = precursors.merge(panel, on=["eFUA_ID", "year"], how="outer")

    out_path = OUT_DIR / "edgar_precursor_emissions_ghs_fua_2012_2022.csv"
    precursors.to_csv(out_path, index=False, encoding="utf-8-sig")

    outputs = merge_precursors(precursors)

    summary_cols = [
        f"edgar_{slug}_emissions_tonnes_per_km2" for slug in POLLUTANTS.values()
    ]
    final_panel_path = outputs.get("complete") or outputs.get("full")
    if final_panel_path:
        final_panel = pd.read_csv(final_panel_path)
        summary = final_panel.groupby("country_iso")[summary_cols].agg(["count", "mean", "min", "max"])
        summary_path = OUT_DIR / "edgar_precursor_emissions_ghs_fua_2012_2022_summary.csv"
        summary.to_csv(summary_path, encoding="utf-8-sig")
    else:
        summary_path = None

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "EDGAR v8.1 Global Air Pollutant Emissions, TOTALS annual NetCDF emissions",
        "pollutants": POLLUTANTS,
        "countries": COUNTRIES,
        "years": list(YEARS),
        "method": "Positive EDGAR 0.1 degree grid-cell centre emissions are summed inside each GHS-FUA polygon. Density divides tonnes by GHS-FUA area in km2.",
        "source_records": source_records,
        "precursor_panel": str(out_path),
        "merged_panel_outputs": outputs,
        "summary_output": str(summary_path) if summary_path else None,
        "rows": int(len(precursors)),
    }
    (META_DIR / "edgar_precursor_emissions_ghs_fua_2012_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Precursor panel: {out_path}")
    for label, path in outputs.items():
        print(f"{label} panel: {path}")
    if summary_path:
        print(f"Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
