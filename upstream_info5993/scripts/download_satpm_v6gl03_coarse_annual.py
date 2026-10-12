from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen, Request


BASE = "https://satpmdata.s3.amazonaws.com"
YEARS = range(1998, 2023)
OUT_DIR = Path("data/raw/pm25/satpm_v6gl03_coarse_annual")
META_DIR = Path("data/metadata")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, out_path: Path) -> dict:
    req = Request(url, headers={"User-Agent": "INFO5993-thesis-data-collector/1.0"})
    with urlopen(req, timeout=120) as response:
        total = int(response.headers.get("Content-Length") or 0)
        tmp_path = out_path.with_suffix(out_path.suffix + ".part")
        downloaded = 0
        with tmp_path.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
        tmp_path.replace(out_path)
    return {"bytes_expected": total, "bytes_downloaded": downloaded}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for year in YEARS:
        key = f"V6GL03/CoarseResolution/GL/Annual/V6GL03.CNNPM25.0p10.GL.{year}01-{year}12.nc"
        url = f"{BASE}/{key}"
        out_path = OUT_DIR / Path(key).name
        if out_path.exists() and out_path.stat().st_size > 0:
            status = "exists"
            size_info = {"bytes_expected": out_path.stat().st_size, "bytes_downloaded": out_path.stat().st_size}
        else:
            print(f"Downloading {year}: {url}", flush=True)
            status = "downloaded"
            size_info = download(url, out_path)
        record = {
            "year": year,
            "status": status,
            "source_url": url,
            "local_path": str(out_path),
            "size_bytes": out_path.stat().st_size,
            "sha256": sha256(out_path),
            **size_info,
        }
        print(json.dumps(record, ensure_ascii=False), flush=True)
        records.append(record)

    metadata = {
        "dataset": "SatPM / ACAG V6.GL.03 coarse-resolution global annual PM2.5",
        "source_bucket": BASE,
        "s3_prefix": "V6GL03/CoarseResolution/GL/Annual/",
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "licence_note": "See SatPM/AWS Open Data Registry source metadata; recorded as CC BY 4.0 in data catalog.",
        "records": records,
    }
    (META_DIR / "satpm_v6gl03_coarse_annual_1998_2022.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
