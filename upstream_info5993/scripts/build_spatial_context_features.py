from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from pyproj import Geod
from shapely.ops import nearest_points


FUA_ZIP = Path(
    "data/raw/boundaries/ghs_fua_r2019a/GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
FUA_LAYER = "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.gpkg"
NATURAL_EARTH_COAST_URL = "https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip"
COAST_ZIP = Path("data/raw/physical/natural_earth/ne_10m_coastline.zip")
INTERIM_DIR = Path("data/interim/spatial_context")
OUT_DIR = Path("data/processed/modelling_panel")
META_DIR = Path("data/metadata")
COUNTRIES = ["CHN", "AUS"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_coastline() -> None:
    COAST_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if COAST_ZIP.exists() and COAST_ZIP.stat().st_size > 0:
        return
    part_path = COAST_ZIP.with_suffix(".zip.part")
    with requests.get(NATURAL_EARTH_COAST_URL, stream=True, timeout=120) as response:
        response.raise_for_status()
        with part_path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    handle.write(chunk)
    part_path.replace(COAST_ZIP)


def safe_divide(num, den):
    if pd.isna(num) or pd.isna(den) or den == 0:
        return np.nan
    return num / den


def load_fuas() -> gpd.GeoDataFrame:
    fua = gpd.read_file(f"zip://{FUA_ZIP}!{FUA_LAYER}")
    fua = fua[fua["Cntry_ISO"].isin(COUNTRIES)].copy()
    fua["eFUA_ID"] = fua["eFUA_ID"].astype(float)
    return fua


def compute_coast_distance(fuas: gpd.GeoDataFrame) -> pd.DataFrame:
    geod = Geod(ellps="WGS84")
    coast = gpd.read_file(f"zip://{COAST_ZIP}")
    coast_wgs84 = coast.to_crs("EPSG:4326")
    coast_metric = coast_wgs84.to_crs("EPSG:3857").reset_index().rename(columns={"index": "coast_id"})

    centroids_moll = gpd.GeoDataFrame(
        fuas[["eFUA_ID", "Cntry_ISO"]].copy(),
        geometry=fuas.geometry.centroid,
        crs=fuas.crs,
    )
    centroids_wgs84 = centroids_moll.to_crs("EPSG:4326")
    centroids_metric = centroids_wgs84.to_crs("EPSG:3857")

    nearest = gpd.sjoin_nearest(
        centroids_metric,
        coast_metric[["coast_id", "geometry"]],
        how="left",
        distance_col="coast_distance_webmerc_m",
    )
    nearest = nearest.drop(columns=["index_right"])
    coast_lookup = coast_wgs84.reset_index().rename(columns={"index": "coast_id"})[
        ["coast_id", "geometry"]
    ]

    rows = []
    for _, row in nearest.to_crs("EPSG:4326").iterrows():
        coast_geom = coast_lookup.loc[coast_lookup["coast_id"] == row["coast_id"], "geometry"].iloc[0]
        point_on_centroid, point_on_coast = nearest_points(row.geometry, coast_geom)
        _, _, distance_m = geod.inv(
            point_on_centroid.x,
            point_on_centroid.y,
            point_on_coast.x,
            point_on_coast.y,
        )
        rows.append(
            {
                "eFUA_ID": row["eFUA_ID"],
                "distance_to_coast_km": distance_m / 1000,
                "is_coastal_50km": int(distance_m / 1000 <= 50),
                "is_coastal_100km": int(distance_m / 1000 <= 100),
            }
        )
    return pd.DataFrame(rows)


def build_features(fuas: gpd.GeoDataFrame, coast_distance: pd.DataFrame) -> pd.DataFrame:
    attrs = fuas.drop(columns="geometry").copy()
    features = attrs[
        [
            "eFUA_ID",
            "eFUA_name",
            "Cntry_ISO",
            "FUA_area",
            "UC_area",
            "FUA_p_2015",
            "UC_p_2015",
            "Com_p_2015",
        ]
    ].rename(
        columns={
            "eFUA_name": "fua_name_check",
            "Cntry_ISO": "country_iso_check",
            "FUA_area": "fua_area_km2_check",
            "UC_area": "urban_centre_area_km2_check",
            "FUA_p_2015": "fua_population_2015_check",
            "UC_p_2015": "urban_centre_population_2015_check",
            "Com_p_2015": "commuting_zone_population_2015_check",
        }
    )

    features["urban_centre_area_share"] = features.apply(
        lambda row: safe_divide(row["urban_centre_area_km2_check"], row["fua_area_km2_check"]),
        axis=1,
    )
    features["urban_centre_population_share"] = features.apply(
        lambda row: safe_divide(
            row["urban_centre_population_2015_check"], row["fua_population_2015_check"]
        ),
        axis=1,
    )
    features["urban_core_density_2015_per_km2"] = features.apply(
        lambda row: safe_divide(
            row["urban_centre_population_2015_check"], row["urban_centre_area_km2_check"]
        ),
        axis=1,
    )
    features["commuting_zone_area_km2"] = (
        features["fua_area_km2_check"] - features["urban_centre_area_km2_check"]
    ).clip(lower=0)
    features["commuting_zone_density_2015_per_km2"] = features.apply(
        lambda row: safe_divide(
            row["commuting_zone_population_2015_check"], row["commuting_zone_area_km2"]
        ),
        axis=1,
    )
    features["log_fua_population_2015"] = np.log1p(features["fua_population_2015_check"])
    features["log_fua_area_km2"] = np.log1p(features["fua_area_km2_check"])
    features["log_population_density_2015_per_km2"] = np.log1p(
        features["fua_population_2015_check"] / features["fua_area_km2_check"]
    )

    centroids = gpd.GeoDataFrame(
        fuas[["eFUA_ID"]].copy(), geometry=fuas.geometry.centroid, crs=fuas.crs
    ).to_crs("EPSG:4326")
    features = features.merge(
        pd.DataFrame(
            {
                "eFUA_ID": centroids["eFUA_ID"],
                "fua_centroid_lon": centroids.geometry.x,
                "fua_centroid_lat": centroids.geometry.y,
            }
        ),
        on="eFUA_ID",
        how="left",
    )
    features = features.merge(coast_distance, on="eFUA_ID", how="left")
    return features


def merge_to_panels(features: pd.DataFrame) -> dict[str, str]:
    outputs = {}
    targets = {
        "extended": OUT_DIR / "extended_all_fua_pm25_edgar_power_ghs_fua_1998_2022.csv",
        "full": OUT_DIR / "full_china_australia_pm25_edgar_power_precursors_ghs_fua_2012_2022.csv",
        "complete": OUT_DIR
        / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power_precursors.csv",
    }
    for label, path in targets.items():
        if not path.exists():
            continue
        panel = pd.read_csv(path)
        duplicate_cols = [
            col
            for col in features.columns
            if col in panel.columns and col not in {"eFUA_ID"}
        ]
        merged = panel.drop(columns=duplicate_cols, errors="ignore").merge(
            features, on="eFUA_ID", how="left"
        )
        if label == "extended":
            out_path = (
                OUT_DIR
                / "extended_all_fua_pm25_edgar_power_spatial_ghs_fua_1998_2022.csv"
            )
        elif label == "full":
            out_path = (
                OUT_DIR
                / "full_china_australia_pm25_edgar_power_precursors_spatial_ghs_fua_2012_2022.csv"
            )
        else:
            out_path = (
                OUT_DIR
                / "analysis_ready_complete_pm25_china_australia_ghs_fua_2012_2022_with_power_precursors_spatial.csv"
            )
        merged.to_csv(out_path, index=False, encoding="utf-8-sig")
        outputs[label] = str(out_path)
    return outputs


def main() -> int:
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    download_coastline()
    fuas = load_fuas()
    coast_distance = compute_coast_distance(fuas)
    features = build_features(fuas, coast_distance)

    features_path = INTERIM_DIR / "ghs_fua_spatial_context_features.csv"
    features.to_csv(features_path, index=False, encoding="utf-8-sig")
    outputs = merge_to_panels(features)

    summary = features.groupby("country_iso_check").agg(
        fua_count=("eFUA_ID", "count"),
        mean_distance_to_coast_km=("distance_to_coast_km", "mean"),
        coastal_50km_share=("is_coastal_50km", "mean"),
        mean_urban_centre_area_share=("urban_centre_area_share", "mean"),
        mean_urban_core_density=("urban_core_density_2015_per_km2", "mean"),
    )
    summary_path = INTERIM_DIR / "ghs_fua_spatial_context_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "boundary_source": str(FUA_ZIP),
        "coastline_source_url": NATURAL_EARTH_COAST_URL,
        "coastline_zip": str(COAST_ZIP),
        "coastline_sha256": sha256_file(COAST_ZIP),
        "method": "Spatial context features derived from GHS-FUA attributes and geodesic centroid-to-coastline distance using Natural Earth 1:10m coastline.",
        "feature_output": str(features_path),
        "summary_output": str(summary_path),
        "merged_panel_outputs": outputs,
        "rows": int(len(features)),
    }
    (META_DIR / "ghs_fua_spatial_context_features.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(summary.to_string())
    print(f"\nFeature output: {features_path}")
    for label, path in outputs.items():
        print(f"{label} panel: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
