from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/processed/modelling_panel/gridcell_monthly_pm25_multisource_2000_2024.csv.gz"
EXTRACTED = ROOT / "data/raw/amlb_dynamic_pm25_monthly_2002_2024/extracted"
OUTPUT = ROOT / "data/interim/amlb_dynamic/amlb_pm25_australia_gridcell_monthly_2002_2024.csv.gz"
MANIFEST = ROOT / "data/interim/amlb_dynamic/amlb_pm25_australia_gridcell_monthly_2002_2024_manifest.json"
RECORD_ID = 19825750


def sample_month(path: Path, cells: pd.DataFrame) -> pd.DataFrame:
    match = re.search(r"(\d{4})(\d{2})\.tif$", path.name)
    if not match:
        raise ValueError(f"Cannot parse year-month from {path.name}")
    with rasterio.open(path) as dataset:
        values = np.fromiter(
            (float(value[0]) for value in dataset.sample(cells[["grid_lon", "grid_lat"]].itertuples(index=False, name=None))),
            dtype=float,
            count=len(cells),
        )
        if dataset.nodata is not None:
            values[np.isclose(values, float(dataset.nodata))] = np.nan
    sampled = cells.copy()
    sampled["year"] = int(match.group(1))
    sampled["month"] = int(match.group(2))
    sampled["amlb_pm25_ug_m3"] = values
    sampled["amlb_source_record"] = f"zenodo:{RECORD_ID}"
    return sampled


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    cells = (
        pd.read_csv(PANEL, usecols=["cell_id", "country_iso", "grid_lon", "grid_lat"])
        .query("country_iso == 'AUS'")
        .drop_duplicates("cell_id")
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    rasters = sorted(EXTRACTED.glob("*/*.tif"))
    if len(rasters) != 270:
        raise RuntimeError(f"Expected 270 AMLB monthly rasters, found {len(rasters)}")

    parts = []
    for sequence, path in enumerate(rasters, start=1):
        parts.append(sample_month(path, cells))
        if sequence % 24 == 0 or sequence == len(rasters):
            print(f"sampled AMLB Australia: {sequence}/{len(rasters)}", flush=True)
    output = pd.concat(parts, ignore_index=True)
    if output.duplicated(["cell_id", "year", "month"]).any():
        raise ValueError("Duplicate AMLB Australian cell-month keys")
    output.to_csv(
        OUTPUT,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": f"https://zenodo.org/records/{RECORD_ID}",
        "doi": "10.5281/zenodo.19825750",
        "country": "Australia",
        "cells": int(output["cell_id"].nunique()),
        "months": int(output[["year", "month"]].drop_duplicates().shape[0]),
        "rows": len(output),
        "coverage": float(output["amlb_pm25_ug_m3"].notna().mean()),
        "output": OUTPUT.relative_to(ROOT).as_posix(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
