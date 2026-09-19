"""Convert VNTraffic MOT ground truth into GRAF inputs.

Reads the VNTraffic clip's MOT-format ground truth and writes:

  data/raw/vntraffic_tracks.jsonl
      One JSON object per detection with keys
      {frame_idx, track_id, bbox_xyxy, confidence, actor_class, class_name}.

  data/raw/vntraffic_homography.yaml
      Scale-only homography so that 1 metre = 1/PPM pixels. TTC and PET
      are scale-invariant so they are correct regardless of PPM; DRAC
      and any absolute-distance threshold inherit the PPM error.

  data/interim/trajectories/vntraffic.json
      Nested world-space trajectories:
      [{track_id, class_name, video_id, frames: [{frame_id, x, y}]}].
      Bottom-centre of the bbox is used as the ground-plane anchor.

Usage:
    python scripts/prepare_vntraffic.py --dataset-root data/external/vntraffic
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

DEFAULT_PPM = 100.0
DEFAULT_VIDEO_ID = "vntraffic"


def convert_tracks(gt_txt: Path, out_jsonl: Path) -> dict:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    track_ids = set()
    max_frame = -1
    with gt_txt.open() as fi, out_jsonl.open("w") as fo:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x, y, bw, bh = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            conf = float(parts[6]) if len(parts) > 6 else 1.0
            fo.write(
                json.dumps(
                    {
                        "frame_idx": frame_idx,
                        "track_id": track_id,
                        "bbox_xyxy": [x, y, x + bw, y + bh],
                        "confidence": conf,
                        "actor_class": "other",
                        "class_name": "other",
                    }
                )
                + "\n"
            )
            n_rows += 1
            track_ids.add(track_id)
            max_frame = max(max_frame, frame_idx)
    return {"rows": n_rows, "tracks": len(track_ids), "frames": max_frame + 1}


def write_homography(out_yaml: Path, ppm: float) -> None:
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    H = [
        [1.0 / ppm, 0.0, 0.0],
        [0.0, 1.0 / ppm, 0.0],
        [0.0, 0.0, 1.0],
    ]
    out_yaml.write_text(yaml.safe_dump({"H": H}))


def write_trajectories(gt_txt: Path, out_json: Path, ppm: float, video_id: str) -> dict:
    tracks = defaultdict(list)
    with gt_txt.open() as fi:
        for line in fi:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame_idx = int(parts[0])
            track_id = int(parts[1])
            x, y, bw, bh = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            cx = x + bw / 2.0
            cy = y + bh
            tracks[track_id].append((frame_idx, cx, cy))

    payload: list[dict[str, Any]] = []
    for tid, rows in sorted(tracks.items()):
        rows.sort()
        payload.append(
            {
                "track_id": tid,
                "class_name": "other",
                "video_id": video_id,
                "frames": [
                    {"frame_id": f, "x": cx / ppm, "y": cy / ppm} for f, cx, cy in rows
                ],
            }
        )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload))
    return {"tracks": len(payload), "rows": sum(len(t["frames"]) for t in payload)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset-root", default="data/external/vntraffic")
    ap.add_argument("--ppm", type=float, default=DEFAULT_PPM)
    ap.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    args = ap.parse_args(argv)

    dataset_root = Path(args.dataset_root)
    gt_txt = dataset_root / "VNTraffic" / "VNTraffic_GroundTruth.txt"
    if not gt_txt.exists():
        print(
            f"ERROR: {gt_txt} not found. Run scripts/fetch_vntraffic.py first.",
            file=sys.stderr,
        )
        return 1

    tracks_out = Path("data/raw/vntraffic_tracks.jsonl")
    homog_out = Path("data/raw/vntraffic_homography.yaml")
    traj_out = Path("data/interim/trajectories/vntraffic.json")

    stats = convert_tracks(gt_txt, tracks_out)
    print(
        f"tracks.jsonl   -> {tracks_out}  "
        f"({stats['rows']} rows, {stats['tracks']} tracks, "
        f"{stats['frames']} frames)"
    )

    write_homography(homog_out, args.ppm)
    print(f"homography     -> {homog_out}  (PPM={args.ppm})")

    tstats = write_trajectories(gt_txt, traj_out, args.ppm, args.video_id)
    print(
        f"trajectories   -> {traj_out}  "
        f"({tstats['tracks']} tracks, {tstats['rows']} rows)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
