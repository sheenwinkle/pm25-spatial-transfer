from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.request import urlopen

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/raw/australia_ground_monitoring/qld/qld_air_quality_station_year_metadata_2015_2024.csv"
RAW_OUTPUT = ROOT / "data/raw/australia_ground_monitoring/qld/qld_air_quality_ckan_metadata_2015_2024.json"
API = "https://www.data.qld.gov.au/api/3/action/package_show?id=air-quality-monitoring-{year}"


def station_from_resource_name(value: str, year: int) -> str:
    value = re.sub(rf"\s+{year}\s+hourly.*$", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"\s*\([^)]*\)\s*$", "", value).strip()
    return value


def main() -> int:
    rows: list[dict[str, object]] = []
    packages: dict[str, object] = {}
    coordinate_pattern = re.compile(
        r"Site\s+location:\*\*\s*Latitude:\s*(-?\d+(?:\.\d+)?)\s*;\s*Longitude:\s*(-?\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )
    for year in range(2015, 2025):
        url = API.format(year=year)
        with urlopen(url, timeout=60) as response:
            payload = json.load(response)
        if not payload.get("success"):
            raise RuntimeError(f"Queensland CKAN request failed for {year}")
        package = payload["result"]
        packages[str(year)] = package
        for resource in package.get("resources", []):
            description = str(resource.get("description") or "")
            match = coordinate_pattern.search(description)
            if not match:
                continue
            resource_name = str(resource.get("name") or "")
            rows.append(
                {
                    "year": year,
                    "station_name": station_from_resource_name(resource_name, year),
                    "station_latitude": float(match.group(1)),
                    "station_longitude": float(match.group(2)),
                    "resource_name": resource_name,
                    "resource_url": resource.get("url"),
                    "package_url": f"https://www.data.qld.gov.au/dataset/air-quality-monitoring-{year}",
                }
            )
    metadata = pd.DataFrame(rows).drop_duplicates(
        ["year", "station_name", "station_latitude", "station_longitude"]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(OUTPUT, index=False)
    RAW_OUTPUT.write_text(json.dumps(packages, indent=2), encoding="utf-8")
    print(
        metadata.groupby("year")
        .agg(resources=("resource_name", "count"), stations=("station_name", "nunique"))
        .to_string()
    )
    print(f"Wrote {len(metadata)} station-year resource rows to {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
