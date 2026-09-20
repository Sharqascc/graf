"""Build world-space trajectories from tracked bounding boxes.

Reads a tracks.jsonl (produced by scripts/run_tracking.py) with fields
video_id, frame_idx, track_id, class_name, confidence, bbox_xyxy;
projects the bottom-center of each bbox into world coordinates using
one of three modes; and writes nested trajectories::

    [{track_id, class_name, video_id, source,
      frames: [{frame_id, x, y}, ...]}]

The three projection modes, in priority order:

* ``--homography_config PATH`` — YAML with an ``H:`` 3x3 matrix. Bottom-
  center pixels are projected via
  :func:`graf.calibration.homography.project_points`.
* ``--pixels_per_meter Z`` — divide pixel coordinates by Z. Use when
  the camera is roughly overhead and a scale factor is known.
* neither given — pass pixels through unchanged and set
  ``source: "pixel"``. Downstream SSM values are then in pixel units,
  not metres. Useful for smoke tests before a homography exists.

The nested output matches what scripts/compute_ssm.py consumes (see its
``_flatten_nested``) and what scripts/prepare_vntraffic.py writes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from graf.calibration.homography import project_points

_REQUIRED_COLUMNS = ("frame_idx", "track_id", "bbox_xyxy")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build world-space trajectories from a tracks.jsonl produced "
            "by scripts/run_tracking.py."
        )
    )
    parser.add_argument(
        "--tracks",
        type=str,
        required=True,
        help="Input tracks.jsonl (run_tracking.py output).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/trajectories",
        help="Directory for trajectories.json.",
    )
    parser.add_argument(
        "--homography_config",
        type=str,
        default=None,
        help="YAML file with an 'H:' 3x3 matrix. Takes priority over "
        "--pixels_per_meter when given.",
    )
    parser.add_argument(
        "--pixels_per_meter",
        type=float,
        default=None,
        help="Scale factor when no homography is available. Ignored if "
        "--homography_config is given.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video fps. Recorded in the output summary.",
    )
    return parser.parse_args(argv)


def load_tracks(path: str | Path) -> list[dict[str, Any]]:
    """Load a tracks.jsonl file into a list of dicts."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"tracks file not found: {path}")
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(json.loads(stripped))
    return rows


def load_homography(path: str | Path) -> np.ndarray:
    """Load a 3x3 homography matrix from a YAML file with an 'H:' key."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"homography config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "H" not in data:
        raise ValueError(f"homography config missing 'H' key: {path}")
    H = np.array(data["H"], dtype=np.float64)
    if H.shape != (3, 3):
        raise ValueError(f"H must be 3x3, got shape {H.shape}")
    return H


def bbox_bottom_center(bbox: list[float]) -> tuple[float, float]:
    """Return the bottom-center pixel of an [x1, y1, x2, y2] bbox.

    Bottom-center is the standard ground-contact point for a bounding
    box under a perspective camera. The centroid would introduce a
    depth-dependent bias in the world projection.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, y2)


def project_points_to_world(
    points: list[tuple[float, float]],
    *,
    homography: np.ndarray | None,
    pixels_per_meter: float | None,
) -> tuple[list[tuple[float, float]], str]:
    """Project pixel points to world coordinates.

    Returns ``(world_points, source_label)`` where ``source_label`` is
    one of ``"homography"``, ``"scale"``, or ``"pixel"``.
    """
    if homography is not None:
        world = project_points(homography, points)
        return [(float(w[0]), float(w[1])) for w in world], "homography"
    if pixels_per_meter is not None:
        if pixels_per_meter <= 0:
            raise ValueError(f"pixels_per_meter must be > 0, got {pixels_per_meter}")
        return [
            (x / pixels_per_meter, y / pixels_per_meter) for x, y in points
        ], "scale"
    return list(points), "pixel"


def build_trajectories(
    rows: list[dict[str, Any]],
    *,
    homography: np.ndarray | None,
    pixels_per_meter: float | None,
) -> list[dict[str, Any]]:
    """Group track rows by (video_id, track_id) and project each frame."""
    if not rows:
        return []

    grouped: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        for col in _REQUIRED_COLUMNS:
            if col not in r:
                raise ValueError(f"track row missing column {col!r}: {r}")
        video_id = str(r.get("video_id", "unknown"))
        grouped[(video_id, r["track_id"])].append(r)

    payload: list[dict[str, Any]] = []
    for (video_id, track_id), frames in grouped.items():
        frames_sorted = sorted(frames, key=lambda r: int(r["frame_idx"]))
        pixel_points = [bbox_bottom_center(r["bbox_xyxy"]) for r in frames_sorted]
        world_points, source = project_points_to_world(
            pixel_points,
            homography=homography,
            pixels_per_meter=pixels_per_meter,
        )
        cls_name = frames_sorted[0].get("class_name", "unknown")
        payload.append(
            {
                "track_id": track_id,
                "class_name": cls_name,
                "video_id": video_id,
                "source": source,
                "frames": [
                    {
                        "frame_id": int(r["frame_idx"]),
                        "x": float(wx),
                        "y": float(wy),
                    }
                    for r, (wx, wy) in zip(frames_sorted, world_points, strict=True)
                ],
            }
        )
    payload.sort(key=lambda t: (t["video_id"], str(t["track_id"])))
    return payload


def main(argv=None) -> int:
    args = parse_args(argv)

    rows = load_tracks(args.tracks)
    homography = None
    if args.homography_config is not None:
        homography = load_homography(args.homography_config)

    trajectories = build_trajectories(
        rows,
        homography=homography,
        pixels_per_meter=args.pixels_per_meter,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "trajectories.json"
    out_path.write_text(json.dumps(trajectories, indent=2), encoding="utf-8")

    total_frames = sum(len(t["frames"]) for t in trajectories)
    source = trajectories[0]["source"] if trajectories else "none"
    print(
        f"Wrote {len(trajectories)} trajectories "
        f"({total_frames} frames, source={source}) to {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
