"""Download and extract the VNTraffic traffic dataset from Zenodo.

Dataset: "Vehicle Tracking" record 18195750.
Contains two short traffic clips (AICC22-Custom, VNTraffic) with
frame-level ground-truth bounding boxes. License: open (Zenodo terms).

Usage:
    python scripts/fetch_vntraffic.py --data-root data/external/vntraffic

By default the archive is removed after extraction; rerunning is a no-op
if the extracted files already exist.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ZENODO_URL = (
    "https://zenodo.org/records/18195750/files/Vehicle_Tracking.zip"
    "?download=1"
)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"cached: {dest}")
        return
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  {dest.stat().st_size / 1e6:.1f} MB")


def extract(zip_path: Path, out_dir: Path) -> None:
    marker = out_dir / "VNTraffic" / "VNTraffic_GroundTruth.txt"
    if marker.exists():
        print(f"already extracted: {marker}")
        return
    print(f"extracting {zip_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out_dir)
    print("  done")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default="data/external/vntraffic")
    ap.add_argument("--keep-zip", action="store_true")
    args = ap.parse_args(argv)

    root = Path(args.data_root)
    zip_path = root / "Vehicle_Tracking.zip"

    download(ZENODO_URL, zip_path)
    extract(zip_path, root)

    if not args.keep_zip:
        zip_path.unlink(missing_ok=True)
        print(f"removed {zip_path} (use --keep-zip to keep)")

    print(f"\ndataset at {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
