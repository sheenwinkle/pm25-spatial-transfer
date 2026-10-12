from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
RAW_DIR = Path("data/raw/meteorology/nasa_power_monthly_tiles_1998_2022")
INTERIM_DIR = Path("data/interim/meteorology")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")

COUNTRIES = ["CHN", "AUS"]
YEARS = range(1998, 2023)
PARAMETERS = ["T2M", "T2M_MAX", "T2M_MIN", "PRECTOTCORR", "WS10M", "RH2M"]
API_URL = "https://power.larc.nasa.gov/api/temporal/monthly/regional"


def load_fuas_with_centroids() -> gpd.GeoDataFrame:
    fua_path = f"zip://{FUA_ZIP}!{FUA_LAYER}"
    fua = gpd.read_file(fua_path)
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    centroids = gpd.GeoSeries(fua.geometry.centroid, crs=fua.crs).to_crs("EPSG:4326")
    fua_wgs84 = fua.to_crs("EPSG:4326")
    fua_wgs84["centroid_lon"] = centroids.x
    fua_wgs84["centroid_lat"] = centroids.y
    fua_wgs84["eFUA_ID"] = fua_wgs84["eFUA_ID"].astype(float)
    return fua_wgs84


def tile_bounds(fuas: gpd.GeoDataFrame) -> list[dict]:
    tiles = []
    for country in COUNTRIES:
        minx, miny, maxx, maxy = fuas[fuas["Cntry_ISO"] == country].total_bounds
        lon_start = math.floor(minx)
        lon_end = math.ceil(maxx)
        lat_start = math.floor(miny)
        lat_end = math.ceil(maxy)
        lon = lon_start
        while lon < lon_end:
            lat = lat_start
            while lat < lat_end:
                tiles.append(
                    {
                        "country_iso": country,
                        "lon_min": lon,
                        "lon_max": min(lon + 10, lon_end),
                        "lat_min": lat,
                        "lat_max": min(lat + 10, lat_end),
                    }
                )
                lat += 10
            lon += 10
    return tiles


def raw_tile_path(parameter: str, tile: dict) -> Path:
    name = (
        f"{parameter}_{tile['country_iso']}_"
        f"lat{tile['lat_min']}_{tile['lat_max']}_"
        f"lon{tile['lon_min']}_{tile['lon_max']}.json"
    )
    return RAW_DIR / name.replace("-", "m")


def download_tile(parameter: str, tile: dict) -> Path:
    out_path = raw_tile_path(parameter, tile)
    if out_path.exists() and out_path.stat().st_size > 0:
        return out_path

    params = {
        "latitude-min": tile["lat_min"],
        "latitude-max": tile["lat_max"],
        "longitude-min": tile["lon_min"],
        "longitude-max": tile["lon_max"],
        "parameters": parameter,
        "community": "SB",
        "start": min(YEARS),
        "end": max(YEARS),
        "format": "JSON",
    }
    for attempt in range(1, 4):
        response = requests.get(API_URL, params=params, timeout=90)
        if response.ok:
            out_path.write_text(response.text, encoding="utf-8")
            time.sleep(0.15)
            return out_path
        if attempt == 3:
            raise RuntimeError(
                f"NASA POWER request failed {response.status_code}: {response.text[:500]}"
            )
        time.sleep(2 * attempt)

    return out_path


def parse_annual_grid(raw_paths: list[Path]) -> pd.DataFrame:
    records = []
    for path in raw_paths:
        parameter = next(
            item for item in sorted(PARAMETERS, key=len, reverse=True) if path.name.startswith(item)
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        for feature in data.get("features", []):
            coords = feature["geometry"]["coordinates"]
            lon, lat = coords[:2]
            elevation = coords[2] if len(coords) > 2 else None
            values = feature["properties"]["parameter"][parameter]
            for year in YEARS:
                key = f"{year}13"
                records.append(
                    {
                        "power_lon": float(lon),
                        "power_lat": float(lat),
                        "power_elevation_m": elevation,
                        "year": year,
                        "parameter": parameter,
                        "value": values.get(key),
                    }
                )

    grid = pd.DataFrame(records)
    grid = grid.drop_duplicates(["power_lon", "power_lat", "year", "parameter"])
    grid = grid.pivot_table(
        index=["power_lon", "power_lat", "power_elevation_m", "year"],
        columns="parameter",
        values="value",
        aggfunc="first",
    ).reset_index()
    grid.columns.name = None
    return grid


def build_fua_to_power_grid_mapping(fuas: gpd.GeoDataFrame, grid: pd.DataFrame) -> pd.DataFrame:
    centroids = gpd.GeoDataFrame(
        fuas[["eFUA_ID", "Cntry_ISO", "centroid_lon", "centroid_lat"]].copy(),
        geometry=gpd.points_from_xy(fuas["centroid_lon"], fuas["centroid_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")
    grid_points = grid[["power_lon", "power_lat"]].drop_duplicates().copy()
    grid_gdf = gpd.GeoDataFrame(
        grid_points,
        geometry=gpd.points_from_xy(grid_points["power_lon"], grid_points["power_lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3857")

    nearest = gpd.sjoin_nearest(
        centroids,
        grid_gdf,
        how="left",
        distance_col="power_centroid_distance_m",
    )
    nearest = pd.DataFrame(nearest.drop(columns=["geometry", "index_right"]))
    nearest["power_centroid_distance_km"] = nearest["power_centroid_distance_m"] / 1000
    return nearest[
        [
            "eFUA_ID",
            "Cntry_ISO",
            "centroid_lon",
            "centroid_lat",
            "power_lon",
            "power_lat",
            "power_centroid_distance_km",
        ]
    ]


def join_to_panels(climate: pd.DataFrame) -> dict[str, str]:
    outputs = {}
    targets = {
        "extended": OUT_DIR / "extended_all_fua_pm25_edgar_ghs_fua_1998_2022.csv",
        "full": OUT_DIR
        / "full_china_australia_pm25_edgar_pm25_ghs_fua_2012_2022.csv",
        "complete": OUT_DIR
        / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022.csv",
    }
    for key, path in targets.items():
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        merged = panel.merge(climate, on=["eFUA_ID", "year"], how="left")
        if key == "extended":
            out_path = OUT_DIR / "extended_all_fua_pm25_edgar_power_ghs_fua_1998_2022.csv"
        elif key == "full":
            out_path = OUT_DIR / "full_china_australia_pm25_edgar_power_ghs_fua_2012_2022.csv"
        else:
            out_path = (
                OUT_DIR
                / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power.csv"
            )
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[key] = str(out_path)
    return outputs


def main() -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    fuas = load_fuas_with_centroids()
    tiles = tile_bounds(fuas)

    raw_paths = []
    total = len(PARAMETERS) * len(tiles)
    count = 0
    for parameter in PARAMETERS:
        for tile in tiles:
            count += 1
            path = download_tile(parameter, tile)
            raw_paths.append(path)
            if count % 25 == 0:
                print(f"Downloaded or cached {count}/{total} NASA POWER tiles")

    grid = parse_annual_grid(raw_paths)
    grid_path = INTERIM_DIR / "nasa_power_annual_grid_china_australia_1998_2022.csv"
    grid.to_csv(grid_path, index=False, encoding="utf-8-sig")

    mapping = build_fua_to_power_grid_mapping(fuas, grid)
    mapping_path = INTERIM_DIR / "ghs_fua_to_nasa_power_grid_mapping.csv"
    mapping.to_csv(mapping_path, index=False, encoding="utf-8-sig")

    climate = mapping.merge(grid, on=["power_lon", "power_lat"], how="left")
    climate = climate.rename(
        columns={
            "T2M": "power_t2m_c",
            "T2M_MAX": "power_t2m_max_c",
            "T2M_MIN": "power_t2m_min_c",
            "PRECTOTCORR": "power_prectotcorr_mm_day",
            "WS10M": "power_ws10m_m_s",
            "RH2M": "power_rh2m_percent",
        }
    )
    climate_path = INTERIM_DIR / "nasa_power_fua_centroid_annual_1998_2022.csv"
    climate.to_csv(climate_path, index=False, encoding="utf-8-sig")

    outputs = join_to_panels(climate)

    coverage = climate.groupby("Cntry_ISO").agg(
        fua_count=("eFUA_ID", "nunique"),
        row_count=("eFUA_ID", "count"),
        t2m_nonmissing=("power_t2m_c", "count"),
        max_centroid_distance_km=("power_centroid_distance_km", "max"),
    )
    coverage_path = INTERIM_DIR / "nasa_power_fua_centroid_coverage_summary_1998_2022.csv"
    coverage.to_csv(coverage_path, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NASA POWER Monthly and Annual API",
        "api_url": API_URL,
        "parameters": PARAMETERS,
        "community": "SB",
        "years": list(YEARS),
        "countries": COUNTRIES,
        "regional_tile_rule": "Tiles are <=10 degrees latitude by <=10 degrees longitude, matching NASA POWER regional API limits.",
        "fua_assignment_method": "Each FUA uses its GHS-FUA geometric centroid assigned to the nearest downloaded NASA POWER meteorology grid point.",
        "raw_tile_count": len(raw_paths),
        "annual_grid_output": str(grid_path),
        "fua_grid_mapping_output": str(mapping_path),
        "fua_annual_climate_output": str(climate_path),
        "panel_outputs": outputs,
        "coverage_summary": str(coverage_path),
    }
    (META_DIR / "nasa_power_fua_centroid_annual_1998_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(coverage.to_string())
    print(f"\nClimate output: {climate_path}")
    for label, path in outputs.items():
        print(f"{label} panel: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
