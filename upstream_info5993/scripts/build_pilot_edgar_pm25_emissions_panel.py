from __future__ import annotations

import json
import re
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
EDGAR_ZIP = Path("data/raw/emissions/edgar_v81_pm25_totals/TOTALS_emi_nc.zip")
EXTRACT_DIR = Path("data/interim/emissions/edgar_v81_pm25_totals_2012_2022")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

YEARS = range(2012, 2023)

SAMPLE_FUA_IDS = {
    10097.0: "Shanghai",
    7704.0: "Beijing",
    9711.0: "Guangzhou",
    1830.0: "Sydney",
    1266.0: "Melbourne",
    2355.0: "Brisbane",
}


def json_safe(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def year_from_name(path: Path | str) -> int:
    match = re.search(r"_(\d{4})_TOTALS_emi\.nc$", str(path))
    if not match:
        raise ValueError(f"Cannot parse year from {path}")
    return int(match.group(1))


def extract_target_years() -> list[Path]:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    extracted_paths: list[Path] = []

    with zipfile.ZipFile(EDGAR_ZIP) as zf:
        names = {
            year_from_name(name): name
            for name in zf.namelist()
            if name.endswith("_TOTALS_emi.nc")
        }
        missing_years = [year for year in YEARS if year not in names]
        if missing_years:
            raise FileNotFoundError(f"Missing EDGAR years in zip: {missing_years}")

        for year in YEARS:
            name = names[year]
            out_path = EXTRACT_DIR / Path(name).name
            if not out_path.exists() or out_path.stat().st_size != zf.getinfo(name).file_size:
                with zf.open(name) as src, out_path.open("wb") as dst:
                    dst.write(src.read())
            extracted_paths.append(out_path)

    return extracted_paths


def points_for_subset(lats, lons, values) -> gpd.GeoDataFrame:
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = float(values[i, j])
            if val > 0:
                rows.append(
                    {
                        "emissions_tonnes": val,
                        "geometry": Point(float(lon), float(lat)),
                    }
                )
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    nc_paths = extract_target_years()

    fua_path = f"zip://{FUA_ZIP}!{FUA_LAYER}"
    fua = gpd.read_file(fua_path)
    sample = fua[fua["eFUA_ID"].isin(SAMPLE_FUA_IDS)].copy()
    sample["sample_label"] = sample["eFUA_ID"].map(SAMPLE_FUA_IDS)
    sample_wgs84 = sample.to_crs("EPSG:4326")

    records = []
    variable_attrs = None
    dataset_attrs = None

    for nc_path in sorted(nc_paths, key=year_from_name):
        year = year_from_name(nc_path)
        ds = xr.open_dataset(nc_path)
        arr = ds["emissions"]
        if variable_attrs is None:
            variable_attrs = json_safe(dict(arr.attrs))
            dataset_attrs = json_safe(dict(ds.attrs))

        for _, row in sample_wgs84.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            sub = arr.sel(
                lon=slice(minx - 0.2, maxx + 0.2),
                lat=slice(miny - 0.2, maxy + 0.2),
            )
            pts = points_for_subset(sub["lat"].values, sub["lon"].values, sub.values)
            if pts.empty:
                total = 0.0
                mean_positive = None
                cell_count = 0
            else:
                joined = gpd.sjoin(
                    pts,
                    gpd.GeoDataFrame([{"geometry": row.geometry}], crs="EPSG:4326"),
                    how="inner",
                    predicate="within",
                )
                total = float(joined["emissions_tonnes"].sum()) if not joined.empty else 0.0
                mean_positive = (
                    float(joined["emissions_tonnes"].mean()) if not joined.empty else None
                )
                cell_count = int(len(joined))

            fua_area_km2 = float(row["FUA_area"])
            records.append(
                {
                    "year": year,
                    "eFUA_ID": row["eFUA_ID"],
                    "fua_name": row["eFUA_name"],
                    "sample_label": row["sample_label"],
                    "country_iso": row["Cntry_ISO"],
                    "country_name": row["Cntry_name"],
                    "fua_area_km2": fua_area_km2,
                    "fua_population_2015": row["FUA_p_2015"],
                    "edgar_pm25_emissions_tonnes": total,
                    "edgar_pm25_emissions_tonnes_per_km2": total / fua_area_km2
                    if fua_area_km2
                    else None,
                    "edgar_positive_grid_cell_mean_tonnes": mean_positive,
                    "edgar_positive_grid_cell_count": cell_count,
                    "edgar_source_file": str(nc_path),
                }
            )

    panel = pd.DataFrame(records).sort_values(["country_iso", "sample_label", "year"])
    emissions_path = OUT_DIR / "pilot_edgar_pm25_emissions_ghs_fua_2012_2022.csv"
    panel.to_csv(emissions_path, index=False, encoding="utf-8-sig")

    summary = (
        panel.groupby(["country_iso", "sample_label"])[
            ["edgar_pm25_emissions_tonnes", "edgar_pm25_emissions_tonnes_per_km2"]
        ]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    summary_path = OUT_DIR / "pilot_edgar_pm25_emissions_ghs_fua_2012_2022_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    pm25_path = OUT_DIR / "pilot_pm25_ghs_fua_2012_2022.csv"
    merged_path = OUT_DIR / "pilot_pm25_edgar_emissions_ghs_fua_2012_2022.csv"
    if pm25_path.exists():
        pm25 = pd.read_csv(pm25_path)
        merged = pm25.merge(
            panel[
                [
                    "year",
                    "eFUA_ID",
                    "edgar_pm25_emissions_tonnes",
                    "edgar_pm25_emissions_tonnes_per_km2",
                    "edgar_positive_grid_cell_mean_tonnes",
                    "edgar_positive_grid_cell_count",
                ]
            ],
            on=["year", "eFUA_ID"],
            how="left",
        )
        merged.to_csv(merged_path, index=False, encoding="utf-8-sig")
    else:
        merged_path = None

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_source": str(FUA_ZIP),
        "edgar_zip": str(EDGAR_ZIP),
        "extracted_dir": str(EXTRACT_DIR),
        "edgar_dataset_attrs": dataset_attrs,
        "edgar_variable_attrs": variable_attrs,
        "method": "Point-in-polygon aggregation over positive EDGAR 0.1 degree grid-cell centres within selected GHS-FUA polygons. Total emissions are summed in tonnes; density divides tonnes by GHS-FUA area in km2.",
        "sample_fua_ids": SAMPLE_FUA_IDS,
        "output_panel": str(emissions_path),
        "output_summary": str(summary_path),
        "output_merged_panel": str(merged_path) if merged_path else None,
        "rows": int(len(panel)),
    }
    (META_DIR / "pilot_edgar_pm25_emissions_ghs_fua_2012_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(panel.head(12).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    if merged_path:
        print(f"\nMerged panel: {merged_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
