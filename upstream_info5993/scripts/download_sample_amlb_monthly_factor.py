from __future__ import annotations

import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio
import requests


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/amlb_dynamic_pm25_monthly_2002_2024"
EXTRACTED = RAW / "extracted"
INTERIM = ROOT / "data/interim/amlb_dynamic"
PANEL = ROOT / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
OUTPUT = INTERIM / "amlb_pm25_gridcell_monthly_2002_2024.csv.gz"
MANIFEST = INTERIM / "amlb_pm25_gridcell_monthly_2002_2024_manifest.json"
RECORD_ID = 19825750
YEARS = range(2002, 2025)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record_files() -> dict[str, dict[str, Any]]:
    response = requests.get(f"https://zenodo.org/api/records/{RECORD_ID}", timeout=60)
    response.raise_for_status()
    return {item["key"]: item for item in response.json()["files"]}


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


def extract(path: Path) -> list[Path]:
    year = int(re.search(r"Y(\d{4})", path.name).group(1))
    target = EXTRACTED / str(year)
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        names = sorted(name for name in archive.namelist() if re.search(r"\d{6}\.tif$", name))
        for name in names:
            destination = target / Path(name).name
            if not destination.exists():
                with archive.open(name) as source, destination.open("wb") as output:
                    output.write(source.read())
    files = sorted(target.glob("*.tif"))
    expected = 6 if year == 2002 else 12
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} monthly rasters for {year}, found {len(files)}")
    return files


def sample(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    match = re.search(r"(\d{4})(\d{2})\.tif$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year-month from {path.name}")
    with rasterio.open(path) as dataset:
        coordinates = cells[["grid_lon", "grid_lat"]].itertuples(index=False, name=None)
        values = np.fromiter(
            (float(item[0]) for item in dataset.sample(coordinates)),
            dtype=float,
            count=len(cells),
        )
        if dataset.nodata is not None:
            values[np.isclose(values, float(dataset.nodata))] = np.nan
    output = cells.copy()
    output["year"] = int(match.group(1))
    output["month"] = int(match.group(2))
    output["amlb_pm25_ug_m3"] = values
    output["amlb_source_record"] = f"zenodo:{RECORD_ID}"
    return output


def main() -> int:
    for path in (RAW, EXTRACTED, INTERIM):
        path.mkdir(parents=True, exist_ok=True)
    files = record_files()
    selected_items: list[dict[str, Any]] = []
    rasters: list[Path] = []
    for year in YEARS:
        key = f"AMLB.Monthly.PM25.Y{year}.zip"
        item = files[key]
        selected_items.append(item)
        rasters.extend(extract(download(item)))

    panel = pd.read_csv(
        PANEL, usecols=["cell_id", "country_iso", "grid_lon", "grid_lat"]
    )
    cells = (
        panel[panel["country_iso"].eq("CHN")]
        .drop_duplicates("cell_id")
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    sampled: list[pd.DataFrame] = []
    for sequence, path in enumerate(sorted(rasters), start=1):
        sampled.append(sample(path, cells))
        print(f"sampled AMLB {path.stem}: {sequence}/{len(rasters)}", flush=True)
    output = pd.concat(sampled, ignore_index=True)
    output.to_csv(OUTPUT, index=False, compression="gzip")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "record": f"https://zenodo.org/records/{RECORD_ID}",
        "doi": "10.5281/zenodo.19825750",
        "years": [min(YEARS), max(YEARS)],
        "months": int(output[["year", "month"]].drop_duplicates().shape[0]),
        "cells": int(output["cell_id"].nunique()),
        "rows": len(output),
        "coverage": float(output["amlb_pm25_ug_m3"].notna().mean()),
        "output": str(OUTPUT.relative_to(ROOT)),
        "files": [
            {
                "key": item["key"],
                "size": int(item["size"]),
                "checksum": item["checksum"],
            }
            for item in selected_items
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
