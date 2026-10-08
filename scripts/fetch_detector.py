"""Download UVH-26 detector weights from HuggingFace.

UVH-26 is an India-specific traffic detection benchmark released by
AIM @ IISc (arXiv:2511.02563): 26,646 surveillance frames from
Bengaluru, 1.8M bounding boxes, 14 fine-grained vehicle classes.
Models trained on it report up to 31.5% higher mAP than COCO-pretrained
baselines on Indian traffic scenes.

Usage:
    python scripts/fetch_detector.py                    # YOLOv11-S (default)
    python scripts/fetch_detector.py --model YOLOv11-X
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/iisc-aim/UVH-26/resolve/main/weights"

VARIANTS = {
    "YOLOv11-S": "UVH-26-MV-YOLOv11-S.pt",
    "YOLOv11-X": "UVH-26-MV-YOLOv11-X.pt",
    "DAMO-YOLO-T": "UVH-26-MV-DAMO-YOLO-T.pt",
    "DAMO-YOLO-L": "UVH-26-MV-DAMO-YOLO-L.pt",
    "RT-DETRv2-S": "UVH-26-MV-RT-DETRv2-S.pt",
    "RT-DETRv2-X": "UVH-26-MV-RT-DETRv2-X.pt",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", choices=list(VARIANTS), default="YOLOv11-S")
    ap.add_argument("--output-dir", default="data/models")
    args = ap.parse_args(argv)

    filename = VARIANTS[args.model]
    url = f"{BASE}/{args.model}/{filename}"
    dst = Path(args.output_dir) / filename
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() and dst.stat().st_size > 1_000_000:
        print(f"cached: {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
        return 0

    print(f"downloading {url}")
    try:
        urllib.request.urlretrieve(url, dst)
    except Exception as e:
        print(f"download failed: {e}", file=sys.stderr)
        return 1
    print(f"  saved {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
