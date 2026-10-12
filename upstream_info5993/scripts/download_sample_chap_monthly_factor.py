from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import xarray as xr


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/chap_pm25_monthly_2019_2024"
EXTRACTED = RAW / "extracted"
INTERIM = ROOT / "data/interim/chap"
PANEL = ROOT / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
OUTPUT = INTERIM / "chap_pm25_gridcell_monthly_2019_2024.csv.gz"
MANIFEST = INTERIM / "chap_pm25_gridcell_monthly_2019_2024_manifest.json"

RECORD_2000_2021 = 6398971
RECORD_2022_PRESENT = 21770406
YEARS = range(2019, 2025)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_files(record_id: int) -> dict[str, dict[str, Any]]:
    response = requests.get(f"https://zenodo.org/api/records/{record_id}", timeout=60)
    response.raise_for_status()
    record = response.json()
    return {item["key"]: item for item in record["files"]}


def download(item: dict[str, Any]) -> Path:
    path = RAW / item["key"]
    expected = str(item["checksum"]).split(":", 1)[1]
    if path.exists() and path.stat().st_size == int(item["size"]) and md5(path) == expected:
        print(f"verified existing {path.name}", flush=True)
        return path
    partial = path.with_suffix(path.suffix + ".part")
    with requests.get(item["links"]["self"], stream=True, timeout=(60, 300)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                if chunk:
                    handle.write(chunk)
    if partial.stat().st_size != int(item["size"]):
        raise RuntimeError(f"Size mismatch for {item['key']}")
    actual = md5(partial)
    if actual != expected:
        raise RuntimeError(f"MD5 mismatch for {item['key']}: {actual} != {expected}")
    partial.replace(path)
    print(f"downloaded and verified {path.name}", flush=True)
    return path


def extract_archive(path: Path) -> list[Path]:
    target = EXTRACTED / path.stem
    target.mkdir(parents=True, exist_ok=True)
    expected = [target / f"CHAP_PM2.5_M1K_{path.stem.split('_')[3]}{month:02d}_V4.nc" for month in range(1, 13)]
    if all(item.exists() for item in expected):
        return expected
    subprocess.run(
        ["tar", "-xf", str(path), "-C", str(target)],
        check=True,
        capture_output=True,
        text=True,
    )
    files = sorted(target.rglob("CHAP_PM2.5_M1K_*.nc"))
    if len(files) != 12:
        raise RuntimeError(f"Expected 12 monthly files in {path.name}, found {len(files)}")
    return files


def parse_month(path: Path) -> tuple[int, int]:
    match = re.search(r"M1K_(\d{4})(\d{2})_V4\.nc$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year-month from {path.name}")
    return int(match.group(1)), int(match.group(2))


def detect_data_array(dataset: xr.Dataset) -> xr.DataArray:
    preferred = ["PM25", "pm25", "PM2.5", "Band1"]
    for name in preferred:
        if name in dataset.data_vars:
            return dataset[name]
    candidates = [dataset[name] for name in dataset.data_vars if dataset[name].ndim >= 2]
    if len(candidates) != 1:
        raise RuntimeError(f"Cannot identify PM2.5 variable from {list(dataset.data_vars)}")
    return candidates[0]


def coordinate_name(array: xr.DataArray, alternatives: list[str]) -> str:
    lower = {name.lower(): name for name in array.coords}
    for alternative in alternatives:
        if alternative.lower() in lower:
            return lower[alternative.lower()]
    raise RuntimeError(f"Cannot identify coordinate among {list(array.coords)}")


def sample_month(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    year, month = parse_month(path)
    with xr.open_dataset(path) as dataset:
        array = detect_data_array(dataset).squeeze(drop=True)
        lon_name = coordinate_name(array, ["lon", "longitude", "x"])
        lat_name = coordinate_name(array, ["lat", "latitude", "y"])
        lons = xr.DataArray(cells["grid_lon"].to_numpy(dtype=float), dims="cell")
        lats = xr.DataArray(cells["grid_lat"].to_numpy(dtype=float), dims="cell")
        values = array.sel(
            {lon_name: lons, lat_name: lats}, method="nearest"
        ).to_numpy().astype(float)
    result = cells.copy()
    result["year"] = year
    result["month"] = month
    result["chap_pm25_ug_m3"] = values
    result["chap_source_record"] = (
        f"zenodo:{RECORD_2000_2021}" if year <= 2021 else f"zenodo:{RECORD_2022_PRESENT}"
    )
    return result


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    INTERIM.mkdir(parents=True, exist_ok=True)

    old = record_files(RECORD_2000_2021)
    new = record_files(RECORD_2022_PRESENT)
    monthly_files: list[Path] = []
    source_items: list[dict[str, Any]] = []
    for year in YEARS:
        if year <= 2021:
            key = f"CHAP_PM2.5_M1K_{year}_V4.rar"
            item = old[key]
            archive = download(item)
            monthly_files.extend(extract_archive(archive))
            source_items.append(item)
        else:
            for month in range(1, 13):
                key = f"CHAP_PM2.5_M1K_{year}{month:02d}_V4.nc"
                item = new[key]
                monthly_files.append(download(item))
                source_items.append(item)

    parsed = {parse_month(path): path for path in monthly_files}
    expected = {(year, month) for year in YEARS for month in range(1, 13)}
    missing = sorted(expected - set(parsed))
    if missing:
        raise RuntimeError(f"Missing CHAP months: {missing}")

    columns = ["cell_id", "country_iso", "grid_lon", "grid_lat"]
    panel = pd.read_csv(PANEL, compression="gzip", usecols=columns)
    cells = (
        panel[panel["country_iso"].eq("CHN")]
        .drop_duplicates("cell_id")
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    sampled: list[pd.DataFrame] = []
    for sequence, key in enumerate(sorted(parsed), start=1):
        sampled.append(sample_month(parsed[key], cells))
        print(f"sampled CHAP {key[0]}-{key[1]:02d}: {sequence}/{len(parsed)}", flush=True)
    output = pd.concat(sampled, ignore_index=True)
    output.to_csv(OUTPUT, index=False, compression="gzip")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            f"https://zenodo.org/records/{RECORD_2000_2021}",
            f"https://zenodo.org/records/{RECORD_2022_PRESENT}",
        ],
        "years": [min(YEARS), max(YEARS)],
        "months": len(parsed),
        "cells": int(cells["cell_id"].nunique()),
        "rows": len(output),
        "output": str(OUTPUT.relative_to(ROOT)),
        "files": [
            {
                "key": item["key"],
                "size": int(item["size"]),
                "checksum": item["checksum"],
            }
            for item in source_items
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
