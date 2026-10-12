from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/chap_pm25_monthly_2019_2024"
NEAREST = ROOT / "data/interim/chap/chap_pm25_gridcell_monthly_2019_2024.csv.gz"
OUTPUT = ROOT / "data/interim/chap/chap_pm25_spatial_support_monthly_2019_2024.csv.gz"
MANIFEST = ROOT / "data/interim/chap/chap_pm25_spatial_support_monthly_2019_2024_manifest.json"
HALF_CELL_DEGREES = 0.05


def parse_month(path: Path) -> tuple[int, int]:
    match = re.search(r"M1K_(\d{4})(\d{2})_V4\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year-month from {path.name}")
    return int(match.group(1)), int(match.group(2))


def monthly_files() -> dict[tuple[int, int], Path]:
    paths = list(RAW.glob("*.nc")) + list((RAW / "extracted").rglob("*.nc"))
    parsed = {parse_month(path): path for path in paths}
    expected = {(year, month) for year in range(2019, 2025) for month in range(1, 13)}
    missing = sorted(expected - set(parsed))
    if missing:
        raise RuntimeError(f"Missing CHAP files: {missing}")
    return {key: parsed[key] for key in sorted(expected)}


def data_array(dataset: xr.Dataset) -> xr.DataArray:
    for name in ("PM2.5", "PM25", "pm25", "Band1"):
        if name in dataset.data_vars:
            return dataset[name].squeeze(drop=True)
    candidates = [dataset[name].squeeze(drop=True) for name in dataset.data_vars]
    candidates = [array for array in candidates if array.ndim == 2]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot identify PM2.5 variable in {list(dataset.data_vars)}")
    return candidates[0]


def coord_name(array: xr.DataArray, alternatives: tuple[str, ...]) -> str:
    names = {name.lower(): name for name in array.coords}
    for candidate in alternatives:
        if candidate in names:
            return names[candidate]
    raise RuntimeError(f"Cannot identify coordinate from {list(array.coords)}")


def window_indices(coordinates: np.ndarray, centres: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lower = np.searchsorted(coordinates, centres - HALF_CELL_DEGREES, side="left")
    upper = np.searchsorted(coordinates, centres + HALF_CELL_DEGREES, side="left")
    widths = upper - lower
    if widths.min() < 9 or widths.max() > 11:
        raise RuntimeError(
            f"Unexpected CHAP window width: min={widths.min()}, max={widths.max()}"
        )
    offsets = np.arange(int(widths.max()))
    indices = lower[:, None] + offsets[None, :]
    valid = offsets[None, :] < widths[:, None]
    return indices, valid


def aggregate_month(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    year, month = parse_month(path)
    with xr.open_dataset(path) as dataset:
        array = data_array(dataset)
        lon_name = coord_name(array, ("lon", "longitude", "x"))
        lat_name = coord_name(array, ("lat", "latitude", "y"))
        array = array.transpose(lat_name, lon_name)
        lons = np.asarray(array[lon_name].values, dtype=float)
        lats = np.asarray(array[lat_name].values, dtype=float)
        values = np.asarray(array.values, dtype=np.float32)

    if lons[0] > lons[-1]:
        lons = lons[::-1]
        values = values[:, ::-1]
    if lats[0] > lats[-1]:
        lats = lats[::-1]
        values = values[::-1, :]

    lon_indices, lon_valid = window_indices(
        lons, cells["grid_lon"].to_numpy(dtype=float)
    )
    lat_indices, lat_valid = window_indices(
        lats, cells["grid_lat"].to_numpy(dtype=float)
    )
    blocks = values[lat_indices[:, :, None], lon_indices[:, None, :]]
    valid_window = lat_valid[:, :, None] & lon_valid[:, None, :]
    blocks = np.where(valid_window, blocks, np.nan)
    finite = np.isfinite(blocks)
    valid_pixels = finite.sum(axis=(1, 2))

    with np.errstate(invalid="ignore"):
        quantiles = np.nanpercentile(blocks, [10, 50, 90], axis=(1, 2))
        result = cells.copy()
        result["year"] = year
        result["month"] = month
        result["chap_pm25_area_mean_ug_m3"] = np.nanmean(blocks, axis=(1, 2))
        result["chap_pm25_area_median_ug_m3"] = quantiles[1]
        result["chap_pm25_area_sd_ug_m3"] = np.nanstd(blocks, axis=(1, 2))
        result["chap_pm25_area_p10_ug_m3"] = quantiles[0]
        result["chap_pm25_area_p90_ug_m3"] = quantiles[2]
        result["chap_pm25_area_valid_pixels"] = valid_pixels
    return result


def main() -> int:
    nearest = pd.read_csv(NEAREST)
    cells = nearest[
        ["cell_id", "country_iso", "grid_lon", "grid_lat"]
    ].drop_duplicates("cell_id").sort_values("cell_id").reset_index(drop=True)
    files = monthly_files()
    frames: list[pd.DataFrame] = []
    for sequence, (key, path) in enumerate(files.items(), start=1):
        frames.append(aggregate_month(path, cells))
        print(f"aggregated CHAP {key[0]}-{key[1]:02d}: {sequence}/{len(files)}", flush=True)

    output = pd.concat(frames, ignore_index=True)
    centre = nearest.rename(columns={"chap_pm25_ug_m3": "chap_pm25_centre_ug_m3"})[
        ["cell_id", "year", "month", "chap_pm25_centre_ug_m3"]
    ]
    output = output.merge(
        centre, on=["cell_id", "year", "month"], how="left", validate="one_to_one"
    )
    output.to_csv(OUTPUT, index=False, compression="gzip")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "spatial_support": "0.1 degree window centred on each SatPM grid cell",
        "half_cell_degrees": HALF_CELL_DEGREES,
        "months": len(files),
        "cells": int(output["cell_id"].nunique()),
        "rows": len(output),
        "valid_pixel_count_min": int(output["chap_pm25_area_valid_pixels"].min()),
        "valid_pixel_count_median": float(output["chap_pm25_area_valid_pixels"].median()),
        "valid_pixel_count_max": int(output["chap_pm25_area_valid_pixels"].max()),
        "output": str(OUTPUT.relative_to(ROOT)),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
