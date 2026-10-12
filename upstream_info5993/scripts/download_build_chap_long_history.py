from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd

import build_chap_spatial_support_features as support
import download_sample_chap_monthly_factor as source


YEARS = range(2000, 2025)
OUTPUT = source.INTERIM / "chap_pm25_spatial_support_monthly_2000_2024.csv.gz"
MANIFEST = source.INTERIM / "chap_pm25_spatial_support_monthly_2000_2024_manifest.json"


def main() -> int:
    for path in (source.RAW, source.EXTRACTED, source.INTERIM):
        path.mkdir(parents=True, exist_ok=True)
    historical = source.record_files(source.RECORD_2000_2021)
    current = source.record_files(source.RECORD_2022_PRESENT)

    source_items = []
    for year in YEARS:
        if year <= 2021:
            key = f"CHAP_PM2.5_M1K_{year}_V4.rar"
            source_items.append(historical[key])
        else:
            for month in range(1, 13):
                key = f"CHAP_PM2.5_M1K_{year}{month:02d}_V4.nc"
                source_items.append(current[key])

    with ThreadPoolExecutor(max_workers=4) as executor:
        downloaded = list(executor.map(source.download, source_items))
    downloaded_by_name = {path.name: path for path in downloaded}

    monthly_files = []
    for year in YEARS:
        if year <= 2021:
            key = f"CHAP_PM2.5_M1K_{year}_V4.rar"
            monthly_files.extend(source.extract_archive(downloaded_by_name[key]))
        else:
            for month in range(1, 13):
                key = f"CHAP_PM2.5_M1K_{year}{month:02d}_V4.nc"
                monthly_files.append(downloaded_by_name[key])

    parsed = {source.parse_month(path): path for path in monthly_files}
    expected = {(year, month) for year in YEARS for month in range(1, 13)}
    missing = sorted(expected - set(parsed))
    if missing:
        raise RuntimeError(f"Missing CHAP months: {missing}")

    panel = pd.read_csv(
        source.PANEL, usecols=["cell_id", "country_iso", "grid_lon", "grid_lat"]
    )
    cells = (
        panel[panel["country_iso"].eq("CHN")]
        .drop_duplicates("cell_id")
        .sort_values("cell_id")
        .reset_index(drop=True)
    )
    frames = []
    for sequence, key in enumerate(sorted(parsed), start=1):
        frames.append(support.aggregate_month(parsed[key], cells))
        print(f"aggregated CHAP {key[0]}-{key[1]:02d}: {sequence}/{len(parsed)}", flush=True)
    output = pd.concat(frames, ignore_index=True)
    output.to_csv(OUTPUT, index=False, compression="gzip")

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": [
            f"https://zenodo.org/records/{source.RECORD_2000_2021}",
            f"https://zenodo.org/records/{source.RECORD_2022_PRESENT}",
        ],
        "spatial_support": "0.1 degree window centred on each SatPM grid cell",
        "years": [min(YEARS), max(YEARS)],
        "months": len(parsed),
        "cells": int(output["cell_id"].nunique()),
        "rows": len(output),
        "valid_pixel_count_median": float(output["chap_pm25_area_valid_pixels"].median()),
        "output": str(OUTPUT.relative_to(source.ROOT)),
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
