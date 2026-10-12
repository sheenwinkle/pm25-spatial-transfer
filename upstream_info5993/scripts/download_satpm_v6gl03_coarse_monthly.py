from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests


START_YEAR = 2000
END_YEAR = 2024
BASE_URL = "https://satpmdata.s3.amazonaws.com"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/raw/pm25/satpm_v6gl03_coarse_monthly"
META_PATH = ROOT / "data/metadata/satpm_v6gl03_coarse_monthly_2000_2024.json"


def item(year: int, month: int) -> tuple[int, int, str, Path]:
    stamp = f"{year}{month:02d}"
    name = f"V6GL03.CNNPM25.0p10.GL.{stamp}-{stamp}.nc"
    url = f"{BASE_URL}/V6GL03/CoarseResolution/GL/Monthly/{year}/{name}"
    return year, month, url, OUT_DIR / name


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_one(year: int, month: int, url: str, path: Path, retries: int = 4) -> dict[str, object]:
    if path.exists() and path.stat().st_size > 1_000_000:
        return {
            "year": year,
            "month": month,
            "status": "existing",
            "source_url": url,
            "local_path": str(path.relative_to(ROOT)),
            "size_bytes": path.stat().st_size,
        }
    last_error = ""
    for attempt in range(1, retries + 1):
        partial = path.with_suffix(path.suffix + ".part")
        try:
            with requests.get(url, stream=True, timeout=180) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            stream.write(chunk)
            if partial.stat().st_size < 1_000_000:
                raise ValueError("downloaded NetCDF was unexpectedly small")
            partial.replace(path)
            return {
                "year": year,
                "month": month,
                "status": "downloaded",
                "source_url": url,
                "local_path": str(path.relative_to(ROOT)),
                "size_bytes": path.stat().st_size,
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            if partial.exists():
                partial.unlink()
            time.sleep(min(2**attempt, 10))
    return {
        "year": year,
        "month": month,
        "status": "failed",
        "source_url": url,
        "local_path": str(path.relative_to(ROOT)),
        "size_bytes": 0,
        "error": last_error,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_PATH.parent.mkdir(parents=True, exist_ok=True)
    items = [item(year, month) for year in range(START_YEAR, END_YEAR + 1) for month in range(1, 13)]
    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(download_one, *entry): entry[:2] for entry in items}
        for completed, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            if completed % 12 == 0 or completed == len(items):
                counts: dict[str, int] = {}
                for row in records:
                    counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
                print(f"SatPM monthly {completed}/{len(items)}: {counts}", flush=True)

    records.sort(key=lambda row: (int(row["year"]), int(row["month"])))
    failures = [record for record in records if record["status"] == "failed"]
    for record in records:
        path = ROOT / str(record["local_path"])
        if path.exists():
            record["sha256"] = sha256(path)
    metadata = {
        "dataset": "SatPM / ACAG V6.GL.03 coarse-resolution global monthly PM2.5",
        "source_bucket": BASE_URL,
        "s3_prefix": "V6GL03/CoarseResolution/GL/Monthly/{year}/",
        "study_period": [START_YEAR, END_YEAR],
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_files": len(items),
        "successful_files": len(items) - len(failures),
        "failed_files": len(failures),
        "total_size_bytes": sum(int(record["size_bytes"]) for record in records),
        "records": records,
    }
    META_PATH.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2), flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
