"""Diagnose a tracking run: is the longest track held correctly, or merged?

Reads a tracks.jsonl produced by scripts/run_tracking.py and reports
per-track movement statistics. The primary diagnostic: for the longest
track, does the bbox center stay put (correct behavior on a stationary
object) or jump around (identity merge)?

This is Phase 0 of docs/sci_tracking_improvement_plan.md. Its output
decides whether Phase 1 (static-object filter) or Phase 2 (association
logic) is the correct first fix.

Usage:
    python scripts/diagnose_tracking.py --tracks X --output_dir Y

Writes:
    <output_dir>/diagnosis.json  -- per-track statistics
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path


@dataclass
class TrackDiag:
    track_id: int
    class_name: str
    n_frames: int
    frame_first: int
    frame_last: int
    cx_first: float
    cy_first: float
    cx_last: float
    cy_last: float
    net_displacement_px: float
    path_length_px: float
    jitter_ratio: float
    bbox_area_median: float
    classification: str


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _center(bbox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def classify_motion(
    net_displacement_px: float,
    jitter_ratio: float,
    stationary_px: float = 5.0,
    jump_ratio: float = 3.0,
) -> str:
    if net_displacement_px > stationary_px:
        return "moving"
    if jitter_ratio >= jump_ratio:
        return "stationary_jumping"
    return "stationary_held"


def diagnose_track(
    track_id: int,
    rows: list[dict],
    stationary_px: float = 5.0,
    jump_ratio: float = 3.0,
) -> TrackDiag:
    rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
    centers = [_center(r["bbox_xyxy"]) for r in rows]
    areas = [_area(r["bbox_xyxy"]) for r in rows]
    classes = Counter(str(r.get("class_name", "unknown")) for r in rows)

    cx_first, cy_first = centers[0]
    cx_last, cy_last = centers[-1]
    net = ((cx_last - cx_first) ** 2 + (cy_last - cy_first) ** 2) ** 0.5

    path = 0.0
    for (x1, y1), (x2, y2) in pairwise(centers):
        path += ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

    jitter = path / max(net, 1.0)
    cls = classify_motion(net, jitter, stationary_px, jump_ratio)

    return TrackDiag(
        track_id=int(track_id),
        class_name=classes.most_common(1)[0][0],
        n_frames=len(rows),
        frame_first=int(rows[0]["frame_idx"]),
        frame_last=int(rows[-1]["frame_idx"]),
        cx_first=float(cx_first),
        cy_first=float(cy_first),
        cx_last=float(cx_last),
        cy_last=float(cy_last),
        net_displacement_px=float(net),
        path_length_px=float(path),
        jitter_ratio=float(jitter),
        bbox_area_median=float(statistics.median(areas)),
        classification=cls,
    )


def diagnose_all(
    tracks: list[dict],
    stationary_px: float = 5.0,
    jump_ratio: float = 3.0,
) -> list[TrackDiag]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for r in tracks:
        by_track[int(r["track_id"])].append(r)
    diags = [
        diagnose_track(tid, rows, stationary_px, jump_ratio)
        for tid, rows in by_track.items()
    ]
    diags.sort(key=lambda d: d.n_frames, reverse=True)
    return diags


def summarise(diags: list[TrackDiag]) -> dict:
    by_class: dict[str, list[TrackDiag]] = defaultdict(list)
    for d in diags:
        by_class[d.class_name].append(d)
    out: dict[str, dict] = {}
    for cls, ds in by_class.items():
        counts = Counter(d.classification for d in ds)
        out[cls] = {
            "n_tracks": len(ds),
            "classification_counts": dict(counts),
            "median_net_displacement_px": statistics.median(
                d.net_displacement_px for d in ds
            ),
            "median_jitter_ratio": statistics.median(d.jitter_ratio for d in ds),
        }
    return out


def interpret(diags: list[TrackDiag]) -> str:
    if not diags:
        return "No tracks in input."
    longest = diags[0]
    lines = [
        f"Longest track {longest.track_id} "
        f"(class={longest.class_name}, n_frames={longest.n_frames}):",
        f"  net_displacement_px: {longest.net_displacement_px:.1f}",
        f"  path_length_px:      {longest.path_length_px:.1f}",
        f"  jitter_ratio:        {longest.jitter_ratio:.1f}",
        f"  classification:      {longest.classification}",
        "",
    ]
    if longest.classification == "stationary_held":
        lines.append(
            "INTERPRETATION: The longest track is a stationary object "
            "held correctly. Phase 1 (static-object filter) is the "
            "fix per docs/sci_tracking_improvement_plan.md."
        )
    elif longest.classification == "stationary_jumping":
        lines.append(
            "INTERPRETATION: The longest track is stationary overall "
            "but its center path is much longer than its net movement. "
            "The tracker is jumping between objects. Phase 2 "
            "(association logic) is the fix per "
            "docs/sci_tracking_improvement_plan.md."
        )
    else:
        lines.append(
            "INTERPRETATION: The longest track is moving. Neither the "
            "static filter nor an obvious association fix applies "
            "directly; inspect the per-frame centers further."
        )
    return chr(10).join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stationary_px", type=float, default=5.0)
    ap.add_argument("--jump_ratio", type=float, default=3.0)
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args(argv)

    track_path = Path(args.tracks)
    if not track_path.exists():
        print(f"error: {track_path} not found", file=sys.stderr)
        return 1

    tracks = _load_jsonl(track_path)
    diags = diagnose_all(tracks, args.stationary_px, args.jump_ratio)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input": str(track_path),
        "n_tracks": len(diags),
        "n_rows": len(tracks),
        "stationary_px": args.stationary_px,
        "jump_ratio": args.jump_ratio,
        "top_tracks": [asdict(d) for d in diags[: args.top_n]],
        "by_class": summarise(diags),
    }
    (out_dir / "diagnosis.json").write_text(json.dumps(report, indent=2) + chr(10))

    print(interpret(diags))
    print()
    print("wrote " + str(out_dir / "diagnosis.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
