"""Draw a metric grid on each frame to QA the homography.

Reads a directory of frames (jpg/png) and a homography YAML, draws a
spacing_m grid on each frame, and writes the result as PNG to
``--output_dir``. The grid lets a human verify at a glance that the
pixel -> world homography lands on the actual road surface.

Usage:
    python scripts/draw_grid.py \
        --frames_dir data/interim/frames/sci/sci \
        --homography_config data/raw/sci_homography.yaml \
        --output_dir data/interim/grid/sci \
        --spacing_m 5.0 \
        --max_frames 10
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2
import numpy as np
import yaml

from graf.calibration.grid_overlay import draw_metric_grid


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--homography_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--spacing_m",
        type=float,
        default=5.0,
        help="Distance between grid lines in metres (default: 5).",
    )
    ap.add_argument(
        "--extent_m",
        type=float,
        default=100.0,
        help="Half-extent of grid coverage in metres (default: 100).",
    )
    ap.add_argument(
        "--max_frames",
        type=int,
        default=0,
        help="Process at most N frames (0 = all). Default: 0.",
    )
    ap.add_argument("--thickness", type=int, default=1)
    args = ap.parse_args(argv)

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"error: frames_dir not found: {frames_dir}", file=sys.stderr)
        return 1

    frames = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    if not frames:
        print(f"error: no jpg/png frames in {frames_dir}", file=sys.stderr)
        return 1
    if args.max_frames > 0:
        frames = frames[: args.max_frames]

    try:
        with Path(args.homography_config).open() as f:
            H = np.array(yaml.safe_load(f)["H"], dtype=float)
    except FileNotFoundError:
        print(
            f"error: homography config not found: {args.homography_config}",
            file=sys.stderr,
        )
        return 1
    except KeyError:
        print(
            f"error: homography config missing 'H' key: {args.homography_config}",
            file=sys.stderr,
        )
        return 1
    if H.shape != (3, 3):
        print(f"error: H is {H.shape}, expected (3, 3)", file=sys.stderr)
        return 1

    try:
        np.linalg.inv(H)
    except np.linalg.LinAlgError:
        print("error: homography is singular (determinant is zero)",
              file=sys.stderr)
        return 1

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    for fp in frames:
        img = cv2.imread(str(fp))
        if img is None:
            print(f"error: could not read {fp}", file=sys.stderr)
            return 1
        out = draw_metric_grid(
            img,
            H,
            spacing_m=args.spacing_m,
            extent_m=args.extent_m,
            thickness=args.thickness,
        )
        out_path = out_dir / f"{fp.stem}_grid.png"
        cv2.imwrite(str(out_path), out)
        written += 1

    print(f"wrote {written} grid frames to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
