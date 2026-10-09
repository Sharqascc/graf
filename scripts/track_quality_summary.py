"""Compute tracking-quality statistics from a detections/tracks pair.

Reads detections.jsonl and tracks.jsonl from a run of run_detection.py
followed by run_tracking.py, and writes a summary.json describing
whether the tracker is stable on this site.

Metrics:

  num_detections, num_track_rows, num_unique_tracks
  track_length_{min,median,max}
  tracks_ge_{5,20,50}_frames       -- long-lived tracks
  pct_tracks_ge_20_frames
  discontinuous_tracks_gap_gt_3    -- tracks with a gap >3 frames
  pct_discontinuous                -- pillar-occlusion / ID-flicker signal
  detections_by_class, tracks_by_class

Usage:
    python scripts/track_quality_summary.py \
        --detections data/interim/detections/sci/detections.jsonl \
        --tracks data/interim/tracks/sci/tracks.jsonl \
        --output data/interim/tracks/sci/summary.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import pairwise
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def summarise(detections: list[dict], tracks: list[dict]) -> dict:
    """Compute tracking-quality statistics. Pure: inputs -> dict."""
    by_track: dict[int, list[int]] = defaultdict(list)
    by_track_class: dict[int, str] = {}
    for r in tracks:
        tid = int(r["track_id"])
        by_track[tid].append(int(r["frame_idx"]))
        by_track_class.setdefault(tid, str(r.get("class_name", "unknown")))

    lengths = [len(v) for v in by_track.values()]

    gaps: list[int] = []
    for fs in by_track.values():
        fs_sorted = sorted(fs)
        if len(fs_sorted) >= 2:
            gaps.append(max(b - a for a, b in pairwise(fs_sorted)))
        else:
            gaps.append(0)
    discontinuous = sum(1 for g in gaps if g > 3)

    detection_classes = Counter(str(d.get("class_name", "unknown")) for d in detections)
    class_counts = Counter(by_track_class.values())

    def _pct(n: int, total: int) -> float:
        return float(100.0 * n / total) if total else 0.0

    ge5 = sum(1 for L in lengths if L >= 5)
    ge20 = sum(1 for L in lengths if L >= 20)
    ge50 = sum(1 for L in lengths if L >= 50)

    return {
        "num_detections": len(detections),
        "num_track_rows": len(tracks),
        "num_unique_tracks": len(by_track),
        "track_length_min": min(lengths) if lengths else 0,
        "track_length_median": statistics.median(lengths) if lengths else 0,
        "track_length_max": max(lengths) if lengths else 0,
        "tracks_ge_5_frames": ge5,
        "tracks_ge_20_frames": ge20,
        "tracks_ge_50_frames": ge50,
        "pct_tracks_ge_20_frames": _pct(ge20, len(lengths)),
        "discontinuous_tracks_gap_gt_3": discontinuous,
        "pct_discontinuous": _pct(discontinuous, len(lengths)),
        "detections_by_class": dict(detection_classes.most_common()),
        "tracks_by_class": dict(class_counts.most_common()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--detections", required=True)
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args(argv)

    det_path = Path(args.detections)
    trk_path = Path(args.tracks)
    if not det_path.exists():
        print(f"error: detections file not found: {det_path}", file=sys.stderr)
        return 1
    if not trk_path.exists():
        print(f"error: tracks file not found: {trk_path}", file=sys.stderr)
        return 1

    summary = summarise(_load_jsonl(det_path), _load_jsonl(trk_path))

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
