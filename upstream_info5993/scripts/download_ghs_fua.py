from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


URL = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/GHSL/"
    "GHS_FUA_UCDB2015_GLOBE_R2019A/V1-0/"
    "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"
)
OUT_DIR = Path("data/raw/boundaries/ghs_fua_r2019a")
META_DIR = Path("data/metadata")
OUT_PATH = OUT_DIR / "GHS_FUA_UCDB2015_GLOBE_R2019A_54009_1K_V1_0.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)
    if not OUT_PATH.exists() or OUT_PATH.stat().st_size == 0:
        req = Request(URL, headers={"User-Agent": "INFO5993-thesis-data-collector/1.0"})
        with urlopen(req, timeout=180) as response:
            total = int(response.headers.get("Content-Length") or 0)
            tmp = OUT_PATH.with_suffix(".zip.part")
            downloaded = 0
            with tmp.open("wb") as f:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
            tmp.replace(OUT_PATH)
    else:
        total = OUT_PATH.stat().st_size
        downloaded = OUT_PATH.stat().st_size

    metadata = {
        "dataset": "GHS-FUA R2019A - GHS functional urban areas, derived from GHS-UCDB R2019A (2015)",
        "source_url": URL,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "local_path": str(OUT_PATH),
        "size_bytes": OUT_PATH.stat().st_size,
        "bytes_expected": total,
        "bytes_downloaded": downloaded,
        "sha256": sha256(OUT_PATH),
        "citation": "Schiavina, Marcello; Moreno-Monroy, Ana; Maffenini, Luca; Veneri, Paolo (2019/2026): GHS-FUA R2019A. European Commission, Joint Research Centre.",
        "doi": "10.2905/347F0337-F2DA-4592-87B3-E25975EC2C95",
    }
    (META_DIR / "ghs_fua_r2019a.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
