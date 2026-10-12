from __future__ import annotations

import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import rasterio
from rasterio.mask import mask


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
STAC_COLLECTION = (
    "https://s3.eu-central-1.wasabisys.com/stac/openlandmap/"
    "nightlights.average_viirs.v21/collection.json"
)
RAW_DIR = Path("data/raw/lights/openlandmap_viirs_v21")
INTERIM_DIR = Path("data/interim/lights")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")
COUNTRIES = ["CHN", "AUS"]
YEARS = range(2012, 2022)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_items() -> dict[int, dict]:
    collection = requests.get(STAC_COLLECTION, timeout=60).json()
    item_links = [link["href"] for link in collection["links"] if link.get("rel") == "item"]
    items = {}
    for href in item_links:
        item_url = requests.compat.urljoin(STAC_COLLECTION, href)
        match = re.search(r"_(\d{4})0101_\d{4}1231", item_url)
        if not match:
            continue
        year = int(match.group(1))
        if year not in YEARS:
            continue
        item = requests.get(item_url, timeout=60).json()
        asset = next(asset for asset in item["assets"].values() if asset.get("main"))
        items[year] = {"item_url": item_url, "asset": asset}
    missing = [year for year in YEARS if year not in items]
    if missing:
        raise FileNotFoundError(f"Missing OpenLandMap VIIRS years: {missing}")
    return items


def download_asset(year: int, asset: dict) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    url = asset["href"]
    out_path = RAW_DIR / Path(url).name
    expected_size = int(asset.get("file:size", 0))
    if out_path.exists() and (expected_size == 0 or out_path.stat().st_size == expected_size):
        return out_path
    part_path = out_path.with_suffix(out_path.suffix + ".part")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part_path.replace(out_path)
    return out_path


def load_fuas() -> gpd.GeoDataFrame:
    fua = gpd.read_file(f"zip://{FUA_ZIP}!{FUA_LAYER}")
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    fua["eFUA_ID"] = fua["eFUA_ID"].astype(float)
    return fua.to_crs("EPSG:4326")


def aggregate_year(tif_path: Path, fuas: gpd.GeoDataFrame, year: int) -> pd.DataFrame:
    rows = []
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        for index, row in fuas.iterrows():
            try:
                arr, _ = mask(src, [row.geometry], crop=True, filled=True)
            except ValueError:
                values = np.array([])
            else:
                values = arr[0].astype("float64").ravel()
                values = values[np.isfinite(values)]
                if nodata is not None:
                    values = values[values != nodata]
                values = values[values >= 0]

            if values.size == 0:
                mean_value = np.nan
                max_value = np.nan
                lit_fraction = np.nan
                pixel_count = 0
            else:
                mean_value = float(values.mean())
                max_value = float(values.max())
                lit_fraction = float((values > 0).mean())
                pixel_count = int(values.size)

            rows.append(
                {
                    "eFUA_ID": row["eFUA_ID"],
                    "year": year,
                    "viirs_ntl_mean": mean_value,
                    "viirs_ntl_max": max_value,
                    "viirs_ntl_lit_fraction": lit_fraction,
                    "viirs_ntl_pixel_count": pixel_count,
                }
            )
    return pd.DataFrame(rows)


def merge_to_panels(nightlights: pd.DataFrame) -> dict[str, str]:
    outputs = {}
    targets = {
        "full_spatial": OUT_DIR
        / "full_china_australia_pm25_edgar_power_precursors_spatial_ghs_fua_2012_2022.csv",
        "complete_spatial": OUT_DIR
        / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power_precursors_spatial.csv",
    }
    for label, path in targets.items():
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        panel_2012_2021 = panel[panel["year"].isin(YEARS)].copy()
        merged = panel_2012_2021.merge(nightlights, on=["eFUA_ID", "year"], how="left")
        if label == "full_spatial":
            out_path = (
                OUT_DIR
                / "full_china_australia_pm25_edgar_power_precursors_spatial_lights_ghs_fua_2012_2021.csv"
            )
        else:
            out_path = (
                OUT_DIR
                / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2021_with_power_precursors_spatial_lights.csv"
            )
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[label] = str(out_path)
    return outputs


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    items = load_items()
    source_records = []
    tif_paths = {}
    for year in YEARS:
        path = download_asset(year, items[year]["asset"])
        tif_paths[year] = path
        source_records.append(
            {
                "year": year,
                "item_url": items[year]["item_url"],
                "asset_url": items[year]["asset"]["href"],
                "local_path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
        print(f"Downloaded or cached VIIRS {year}: {path.name}")

    fuas = load_fuas()
    yearly = []
    for year, path in tif_paths.items():
        print(f"Aggregating VIIRS nightlights {year}")
        yearly.append(aggregate_year(path, fuas, year))

    nightlights = pd.concat(yearly, ignore_index=True)
    nightlights_path = INTERIM_DIR / "openlandmap_viirs_v21_ghs_fua_2012_2021.csv"
    nightlights.to_csv(nightlights_path, index=False, encoding="utf-8-sig")
    outputs = merge_to_panels(nightlights)

    final_panel_path = outputs.get("complete_spatial")
    if final_panel_path:
        final_panel = pd.read_csv(final_panel_path)
        summary = final_panel.groupby("country_iso").agg(
            rows=("eFUA_ID", "count"),
            fuas=("eFUA_ID", "nunique"),
            ntl_nonmissing=("viirs_ntl_mean", "count"),
            ntl_mean=("viirs_ntl_mean", "mean"),
            ntl_lit_fraction=("viirs_ntl_lit_fraction", "mean"),
        )
    else:
        summary = nightlights.agg(
            ntl_nonmissing=("viirs_ntl_mean", "count"),
            ntl_mean=("viirs_ntl_mean", "mean"),
        )
    summary_path = INTERIM_DIR / "openlandmap_viirs_v21_ghs_fua_coverage_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenLandMap static STAC VIIRS annual nighttime lights v2.1",
        "stac_collection": STAC_COLLECTION,
        "years": list(YEARS),
        "method": "Zonal raster summary of 500 m VIIRS annual nighttime light values within GHS-FUA polygons. Outputs mean, max, lit fraction, and pixel count.",
        "source_records": source_records,
        "nightlights_output": str(nightlights_path),
        "summary_output": str(summary_path),
        "merged_panel_outputs": outputs,
        "note": "This enhancement covers 2012-2021. It is not forced into the 2022 main panel because this OpenLandMap v2.1 source does not provide 2022.",
    }
    (META_DIR / "openlandmap_viirs_v21_ghs_fua_2012_2021.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string())
    print(f"\nNightlights output: {nightlights_path}")
    for label, path in outputs.items():
        print(f"{label} panel: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
