"""Greedy-IoU multi-frame tracker over per-frame detections.

Reads a detection JSONL (see ``scripts/run_detection.py``) and produces a
tracked JSONL by associating detections across frames with a greedy
IoU-match in the same class. Not the real ByteTrack — see
``configs/tracking/bytetrack.yaml`` for the naming caveat.

Config keys honoured:

    track_thresh   minimum detection confidence to consider
    iou_threshold  IoU threshold for association
    track_buffer   frames without a match before a track is dropped
    min_box_area   minimum bbox area (px^2)

CLI flags override config values. Detections below ``track_thresh`` or
below ``min_box_area`` are ignored; matches within a stale track are not
considered.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "tracking" / "bytetrack.yaml"
)


def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def load_config(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tracking config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def resolve(cfg: dict, key: str, cli_value, default):
    if cli_value is not None:
        return cli_value
    value = cfg.get(key)
    return default if value is None else value


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Track detections across frames with greedy IoU association."
    )
    p.add_argument("--detections", required=True, help="Input detections JSONL")
    p.add_argument(
        "--output_dir", required=True, help="Output directory for tracks.jsonl"
    )
    p.add_argument(
        "--config", default=str(_DEFAULT_CONFIG), help="Tracking config YAML"
    )
    p.add_argument(
        "--track_thresh",
        type=float,
        default=None,
        help="Override config 'track_thresh'",
    )
    p.add_argument(
        "--iou_threshold",
        type=float,
        default=None,
        help="Override config 'iou_threshold'",
    )
    p.add_argument(
        "--track_buffer", type=int, default=None, help="Override config 'track_buffer'"
    )
    p.add_argument(
        "--min_box_area",
        type=float,
        default=None,
        help="Override config 'min_box_area'",
    )
    return p.parse_args(argv)


def box_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def track(
    detections: list[dict],
    *,
    track_thresh: float,
    iou_threshold: float,
    track_buffer: int,
    min_box_area: float,
) -> list[dict]:
    """Run greedy IoU tracking; return per-frame track rows."""
    # Filter by confidence and box area up front.
    detections = [
        d
        for d in detections
        if d.get("confidence", 1.0) >= track_thresh
        and box_area(d["bbox_xyxy"]) >= min_box_area
    ]

    by_frame: dict[int, list[dict]] = defaultdict(list)
    for det in detections:
        by_frame[det["frame_idx"]].append(det)

    tracks: list[dict] = []
    active_tracks: dict[int, dict] = {}
    next_id = 0

    for frame_idx in sorted(by_frame):
        dets = by_frame[frame_idx]
        unmatched = list(range(len(dets)))

        # Try to match each active track to an unmatched detection.
        for tid, info in list(active_tracks.items()):
            if not unmatched:
                break
            best_iou = 0.0
            best_det = -1
            for i in unmatched:
                det = dets[i]
                if det["class_name"] != info["class_name"]:
                    continue
                score = iou(info["bbox_xyxy"], det["bbox_xyxy"])
                if score > best_iou:
                    best_iou = score
                    best_det = i
            if best_det >= 0 and best_iou >= iou_threshold:
                det = dets[best_det]
                active_tracks[tid] = {
                    "bbox_xyxy": det["bbox_xyxy"],
                    "class_name": det["class_name"],
                    "frames_since_update": 0,
                }
                tracks.append(
                    {
                        "video_id": det["video_id"],
                        "frame_idx": det["frame_idx"],
                        "track_id": tid,
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox_xyxy": det["bbox_xyxy"],
                    }
                )
                unmatched.remove(best_det)

        # New tracks for detections not matched to any active track.
        for i in unmatched:
            det = dets[i]
            tid = next_id
            next_id += 1
            active_tracks[tid] = {
                "bbox_xyxy": det["bbox_xyxy"],
                "class_name": det["class_name"],
                "frames_since_update": 0,
            }
            tracks.append(
                {
                    "video_id": det["video_id"],
                    "frame_idx": det["frame_idx"],
                    "track_id": tid,
                    "class_name": det["class_name"],
                    "confidence": det["confidence"],
                    "bbox_xyxy": det["bbox_xyxy"],
                }
            )

        # Increment staleness for tracks NOT matched this frame, then drop
        # those past the buffer. This is the piece the previous version was
        # missing — frames_since_update was set to 0 on every update path
        # and never incremented, so the buffer never triggered.
        matched_tids = {t["track_id"] for t in tracks if t["frame_idx"] == frame_idx}
        for tid in list(active_tracks.keys()):
            if tid not in matched_tids:
                active_tracks[tid]["frames_since_update"] += 1
        stale = [
            tid
            for tid, info in active_tracks.items()
            if info["frames_since_update"] > track_buffer
        ]
        for tid in stale:
            del active_tracks[tid]

    return tracks


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    track_thresh = resolve(cfg, "track_thresh", args.track_thresh, 0.5)
    iou_threshold = resolve(cfg, "iou_threshold", args.iou_threshold, 0.3)
    track_buffer = int(resolve(cfg, "track_buffer", args.track_buffer, 30))
    min_box_area = resolve(cfg, "min_box_area", args.min_box_area, 10.0)

    with open(args.detections) as f:
        detections = [json.loads(line) for line in f if line.strip()]

    tracks = track(
        detections,
        track_thresh=track_thresh,
        iou_threshold=iou_threshold,
        track_buffer=track_buffer,
        min_box_area=min_box_area,
    )

    out_path = Path(args.output_dir) / "tracks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tracks:
            f.write(json.dumps(t) + "\n")
    print(f"Tracks saved to {out_path} ({len(tracks)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
