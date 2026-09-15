"""Multi-frame tracker over per-frame detections with motion prediction.

Reads a detection JSONL (see ``scripts/run_detection.py``) and produces a
tracked JSONL by associating detections across frames with a greedy IoU
match in the same class. Not the real ByteTrack — see
``configs/tracking/bytetrack.yaml`` for the naming caveat.

Motion prediction
-----------------
When ``use_motion_prediction`` is on, each active track keeps an
exponentially-smoothed velocity (in px/frame). Before matching at frame
*F*, the track's last-known bbox is shifted by
``velocity * (F - last_matched_frame)`` (capped per-axis at
``max_prediction_offset_px``). IoU is computed against the shifted bbox,
but the *real* matched detection's bbox is what gets emitted. This fixes
the stride-artifact where a fast mover's bbox jumps past the IoU
threshold between processed frames and spawns a new track every frame.

Config keys honoured
--------------------
    track_thresh               min detection confidence
    iou_threshold              greedy IoU association threshold
    track_buffer               frames without a match before a track is dropped
    min_box_area               min bbox area (px^2)
    use_motion_prediction      bool; enable velocity-compensated IoU
    velocity_smoothing         EMA alpha in (0, 1]; 1.0 = latest velocity
    max_prediction_offset_px   per-axis cap on the predicted shift

CLI flags override config values.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "tracking" / "bytetrack.yaml"
)

# Type alias for bbox (x1, y1, x2, y2)
BBox = tuple[float, float, float, float]


def iou(boxA: BBox, boxB: BBox) -> float:
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    boxAArea = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    boxBArea = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom


def box_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def shift_bbox(bbox: BBox, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)


def centroid_distance(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    return math.hypot(ax - bx, ay - by)


def association_score(
    probe: BBox,
    det: BBox,
    max_centroid_distance_px: float,
) -> tuple[float, float]:
    """Return (score, iou).

    score = max(iou, 1 - centroid_distance / max_centroid_distance_px),
    clipped to [0, 1]. When max_centroid_distance_px is non-positive the
    centroid term is disabled and score == iou.
    """
    iou_score = iou(probe, det)
    if max_centroid_distance_px <= 0:
        return iou_score, iou_score
    cdist = centroid_distance(probe, det)
    centroid_score = max(0.0, 1.0 - cdist / max_centroid_distance_px)
    return max(iou_score, centroid_score), iou_score


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
        description=(
            "Track detections across frames with greedy IoU association "
            "and optional motion prediction."
        )
    )
    p.add_argument("--detections", required=True, help="Input detections JSONL")
    p.add_argument(
        "--output_dir", required=True, help="Output directory for tracks.jsonl"
    )
    p.add_argument(
        "--config", default=str(_DEFAULT_CONFIG), help="Tracking config YAML"
    )
    p.add_argument("--track_thresh", type=float, default=None)
    p.add_argument("--iou_threshold", type=float, default=None)
    p.add_argument("--track_buffer", type=int, default=None)
    p.add_argument("--min_box_area", type=float, default=None)
    p.add_argument(
        "--no_motion_prediction",
        action="store_true",
        help="Disable velocity-compensated matching",
    )
    p.add_argument("--velocity_smoothing", type=float, default=None)
    p.add_argument("--max_prediction_offset_px", type=float, default=None)
    p.add_argument("--max_centroid_distance_px", type=float, default=None)
    return p.parse_args(argv)


def _make_track_state(bbox: BBox, class_name: str, frame_idx: int) -> dict:
    cx, cy = bbox_center(bbox)
    return {
        "bbox_xyxy": bbox,
        "class_name": class_name,
        "centroid": (cx, cy),
        "velocity": (0.0, 0.0),
        "last_matched_frame": int(frame_idx),
        "frames_since_update": 0,
    }


def _update_track_state(
    state: dict,
    bbox: BBox,
    frame_idx: int,
    velocity_smoothing: float,
) -> None:
    new_cx, new_cy = bbox_center(bbox)
    old_cx, old_cy = state["centroid"]
    delta_frames = max(1, int(frame_idx) - int(state["last_matched_frame"]))
    raw_vx = (new_cx - old_cx) / delta_frames
    raw_vy = (new_cy - old_cy) / delta_frames
    prev_vx, prev_vy = state["velocity"]
    alpha = velocity_smoothing
    state["velocity"] = (
        alpha * raw_vx + (1.0 - alpha) * prev_vx,
        alpha * raw_vy + (1.0 - alpha) * prev_vy,
    )
    state["centroid"] = (new_cx, new_cy)
    state["bbox_xyxy"] = bbox
    state["last_matched_frame"] = int(frame_idx)
    state["frames_since_update"] = 0


def _predicted_bbox(
    state: dict,
    frame_idx: int,
    max_offset: float,
) -> BBox:
    delta = int(frame_idx) - int(state["last_matched_frame"])
    if delta <= 0:
        return state["bbox_xyxy"]
    vx, vy = state["velocity"]
    dx = max(-max_offset, min(max_offset, vx * delta))
    dy = max(-max_offset, min(max_offset, vy * delta))
    return shift_bbox(state["bbox_xyxy"], dx, dy)


def track(
    detections: list[dict],
    *,
    track_thresh: float,
    iou_threshold: float,
    track_buffer: int,
    min_box_area: float,
    use_motion_prediction: bool = True,
    velocity_smoothing: float = 0.5,
    max_prediction_offset_px: float = 300.0,
    max_centroid_distance_px: float = 200.0,
) -> list[dict]:
    """Run motion-aware greedy IoU tracking; return per-frame track rows."""
    if not 0.0 < velocity_smoothing <= 1.0:
        raise ValueError(
            f"velocity_smoothing must be in (0, 1], got {velocity_smoothing}"
        )
    if max_prediction_offset_px < 0:
        raise ValueError(
            f"max_prediction_offset_px must be >= 0, got {max_prediction_offset_px}"
        )
    if max_centroid_distance_px < 0:
        raise ValueError(
            f"max_centroid_distance_px must be >= 0, got {max_centroid_distance_px}"
        )

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
        matched_this_frame: set[int] = set()

        for tid, state in list(active_tracks.items()):
            if not unmatched:
                break
            if use_motion_prediction:
                probe_bbox = _predicted_bbox(state, frame_idx, max_prediction_offset_px)
            else:
                probe_bbox = state["bbox_xyxy"]

            best_score = 0.0
            best_det = -1
            for i in unmatched:
                det = dets[i]
                if det["class_name"] != state["class_name"]:
                    continue
                score, _ = association_score(
                    probe_bbox, det["bbox_xyxy"], max_centroid_distance_px
                )
                if score > best_score:
                    best_score = score
                    best_det = i
            if best_det >= 0 and best_score >= iou_threshold:
                det = dets[best_det]
                _update_track_state(
                    state, det["bbox_xyxy"], frame_idx, velocity_smoothing
                )
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
                matched_this_frame.add(tid)

        # New tracks for unmatched detections.
        for i in unmatched:
            det = dets[i]
            tid = next_id
            next_id += 1
            active_tracks[tid] = _make_track_state(
                det["bbox_xyxy"], det["class_name"], frame_idx
            )
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

        # Staleness: unmatched tracks age out, matched ones reset to 0.
        for tid, state in active_tracks.items():
            if tid not in matched_this_frame and int(
                state["last_matched_frame"]
            ) != int(frame_idx):
                state["frames_since_update"] += 1
        stale = [
            tid
            for tid, state in active_tracks.items()
            if state["frames_since_update"] > track_buffer
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

    if args.no_motion_prediction:
        use_motion_prediction = False
    else:
        use_motion_prediction = bool(resolve(cfg, "use_motion_prediction", None, True))
    velocity_smoothing = float(
        resolve(cfg, "velocity_smoothing", args.velocity_smoothing, 0.5)
    )
    max_prediction_offset_px = float(
        resolve(
            cfg,
            "max_prediction_offset_px",
            args.max_prediction_offset_px,
            300.0,
        )
    )
    max_centroid_distance_px = float(
        resolve(
            cfg,
            "max_centroid_distance_px",
            args.max_centroid_distance_px,
            200.0,
        )
    )
    # Coupling: disabling motion prediction disables the centroid fallback
    # too, unless the caller explicitly set max_centroid_distance_px. This
    # preserves the old IoU-only behavior (including its stride artifact)
    # when a user asks for it.
    if not use_motion_prediction and args.max_centroid_distance_px is None:
        max_centroid_distance_px = 0.0

    with open(args.detections) as f:
        detections = [json.loads(line) for line in f if line.strip()]

    tracks = track(
        detections,
        track_thresh=track_thresh,
        iou_threshold=iou_threshold,
        track_buffer=track_buffer,
        min_box_area=min_box_area,
        use_motion_prediction=use_motion_prediction,
        velocity_smoothing=velocity_smoothing,
        max_prediction_offset_px=max_prediction_offset_px,
        max_centroid_distance_px=max_centroid_distance_px,
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
