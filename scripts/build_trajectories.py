from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build toy trajectories for GRAF.")
    parser.add_argument("--outdir", type=str, default="outputs/trajectories", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trajectories = [
        {
            "track_id": 1,
            "class_name": "car",
            "frames": [
                {"frame_id": 1, "x": 0.0, "y": 0.0},
                {"frame_id": 2, "x": 1.0, "y": 0.0},
                {"frame_id": 3, "x": 2.0, "y": 0.0},
            ],
        },
        {
            "track_id": 2,
            "class_name": "pedestrian",
            "frames": [
                {"frame_id": 1, "x": 5.0, "y": 1.0},
                {"frame_id": 2, "x": 5.2, "y": 1.0},
                {"frame_id": 3, "x": 5.4, "y": 1.0},
            ],
        },
    ]

    out_path = outdir / "trajectories.json"
    out_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")
    print(f"Wrote trajectories to {out_path}")


if __name__ == "__main__":
    main()
