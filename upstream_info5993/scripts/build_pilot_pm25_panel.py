from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import xarray as xr
from shapely.geometry import Point


FUA_ZIP = Path("data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip")
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
PM25_DIR = Path("data/raw/pm25/satpm_v6gl03_coarse_annual")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

SAMPLE_FUA_IDS = {
    10097.0: "Shanghai",
    7704.0: "Beijing",
    9711.0: "Guangzhou",
    1830.0: "Sydney",
    1266.0: "Melbourne",
    2355.0: "Brisbane",
}


def year_from_name(path: Path) -> int:
    match = re.search(r"\.(\d{4})01-\d{4}12\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year from {path.name}")
    return int(match.group(1))


def points_for_subset(lats, lons, values) -> gpd.GeoDataFrame:
    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            val = float(values[i, j])
            if val >= 0:
                rows.append({"pm25": val, "geometry": Point(float(lon), float(lat))})
    return gpd.GeoDataFrame(rows, crs="EPSG:4326")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    fua_path = f"zip://{FUA_ZIP}!{FUA_LAYER}"
    fua = gpd.read_file(fua_path)
    sample = fua[fua["eFUA_ID"].isin(SAMPLE_FUA_IDS)].copy()
    sample["sample_label"] = sample["eFUA_ID"].map(SAMPLE_FUA_IDS)
    sample_wgs84 = sample.to_crs("EPSG:4326")

    # Save exact pilot boundary attributes for audit.
    sample.drop(columns="geometry").to_csv(
        "data/interim/ghs_fua_pilot_attributes.csv", index=False, encoding="utf-8-sig"
    )
    sample_wgs84.to_file(
        "data/interim/ghs_fua_pilot_boundaries.geojson", driver="GeoJSON"
    )

    records = []
    for nc_path in sorted(PM25_DIR.glob("V6GL03.CNNPM25.0p10.GL.*.nc")):
        year = year_from_name(nc_path)
        if year < 2012 or year > 2022:
            continue
        ds = xr.open_dataset(nc_path)
        arr = ds["PM25"]

        for _, row in sample_wgs84.iterrows():
            minx, miny, maxx, maxy = row.geometry.bounds
            # Slight buffer includes edge grid-cell centres.
            sub = arr.sel(
                lon=slice(minx - 0.2, maxx + 0.2),
                lat=slice(miny - 0.2, maxy + 0.2),
            )
            pts = points_for_subset(sub["lat"].values, sub["lon"].values, sub.values)
            if pts.empty:
                pm25_mean = None
                cell_count = 0
            else:
                joined = gpd.sjoin(
                    pts,
                    gpd.GeoDataFrame([{"geometry": row.geometry}], crs="EPSG:4326"),
                    how="inner",
                    predicate="within",
                )
                pm25_mean = float(joined["pm25"].mean()) if not joined.empty else None
                cell_count = int(len(joined))

            records.append(
                {
                    "year": year,
                    "eFUA_ID": row["eFUA_ID"],
                    "fua_name": row["eFUA_name"],
                    "sample_label": row["sample_label"],
                    "country_iso": row["Cntry_ISO"],
                    "country_name": row["Cntry_name"],
                    "fua_area_km2": row["FUA_area"],
                    "fua_population_2015": row["FUA_p_2015"],
                    "pm25_mean_ug_m3": pm25_mean,
                    "pm25_grid_cell_count": cell_count,
                    "pm25_source_file": str(nc_path),
                }
            )

    panel = pd.DataFrame(records).sort_values(["country_iso", "sample_label", "year"])
    out_path = OUT_DIR / "pilot_pm25_ghs_fua_2012_2022.csv"
    panel.to_csv(out_path, index=False, encoding="utf-8-sig")

    summary = (
        panel.groupby(["country_iso", "sample_label"])["pm25_mean_ug_m3"]
        .agg(["count", "mean", "min", "max"])
        .reset_index()
    )
    summary_path = OUT_DIR / "pilot_pm25_ghs_fua_2012_2022_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_source": str(FUA_ZIP),
        "pm25_source_dir": str(PM25_DIR),
        "method": "Point-in-polygon mean over 0.1 degree SatPM grid-cell centres within selected GHS-FUA polygons. PM25 values < 0 treated as missing.",
        "sample_fua_ids": SAMPLE_FUA_IDS,
        "output_panel": str(out_path),
        "output_summary": str(summary_path),
        "rows": int(len(panel)),
    }
    (META_DIR / "pilot_pm25_ghs_fua_2012_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(panel.head(12).to_string(index=False))
    print("\nSummary:")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
