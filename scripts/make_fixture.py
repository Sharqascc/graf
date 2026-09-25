"""Generate the synthetic fixture used by the end-to-end smoke test.

Produces a small, deterministic set of tracks, a homography, and the
corresponding per-frame PyG graphs. Output is byte-identical across
runs on any machine (given the same numpy version), so the fixture can
be committed to the repo and inspected by reviewers.

The tracks JSONL matches the schema `graf.training.conflict_pairs`
expects: `frame_idx, track_id, confidence, bbox_xyxy, actor_class`.
World coordinates are derived from `bbox_xyxy` via the homography, so
the fixture uses pixel coordinates equal to world coordinates (identity
homography).

Usage:
    python scripts/make_fixture.py --out data/fixtures/demo
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np
import yaml

SEED = 20260925
NUM_FRAMES = 60
FPS = 30.0

# A small mixed-traffic scene. Actors 1 and 2 are cars closing head-on
# with a 2 m lateral offset, so they pass each other rather than collide.
# With x0=+-2 and vx=+-1.5, their separation drops below the 3 m
# distance threshold from about frame 18 and their TTC stays under 1.5 s,
# producing a run of critical frames long enough to satisfy the
# sustained label at run_length=3. The remaining four actors are moving
# away and act as noise.
ACTORS = [
    # (track_id, actor_class, x0, y0, vx, vy, width, height)
    (1, "car", -2.0, -1.0, 1.5, 0.0, 2.0, 4.5),
    (2, "car", 2.0, 1.0, -1.5, 0.0, 2.0, 4.5),
    (3, "pedestrian", 0.0, 20.0, 0.0, -0.5, 0.6, 1.7),
    (4, "two_wheeler", 0.0, -20.0, 0.0, 0.5, 0.8, 1.9),
    (5, "bicycle", 15.0, 10.0, -0.3, -0.2, 0.7, 1.8),
    (6, "auto_rickshaw", -15.0, 5.0, 0.4, -0.3, 1.5, 2.5),
]

CONFIDENCE = 0.9


def make_tracks() -> list[dict]:
    """One JSONL record per (frame, actor). Bbox bottom-centre equals the
    actor's world position because the homography is identity and the
    bbox is centred on (x, y - h/2)."""
    rows: list[dict] = []
    for frame_idx in range(NUM_FRAMES):
        t = frame_idx / FPS
        for track_id, cls, x0, y0, vx, vy, w, h in ACTORS:
            x = x0 + vx * t
            y = y0 + vy * t
            # bbox in pixel coords equal to world coords (identity H)
            x1 = x - w / 2.0
            x2 = x + w / 2.0
            y1 = y - h
            y2 = y
            rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "track_id": int(track_id),
                    "confidence": CONFIDENCE,
                    "bbox_xyxy": [
                        round(x1, 4),
                        round(y1, 4),
                        round(x2, 4),
                        round(y2, 4),
                    ],
                    "actor_class": cls,
                    "frame_w": 1920,
                    "frame_h": 1080,
                }
            )
    return rows


def make_homography() -> dict:
    return {"H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}


def write_tracks(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def write_homography(h: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(h, f, sort_keys=True)


def _call_builder(rows: list[dict], frame_id: int, video_id: str):
    """Call build_pyg_graph_for_frame with whatever signature the repo has."""
    from graf.graph.builders import build_pyg_graph_for_frame

    attempts = [
        lambda: build_pyg_graph_for_frame(rows, frame_id=frame_id, video_id=video_id),
        lambda: build_pyg_graph_for_frame(rows, frame_id, video_id),
        lambda: build_pyg_graph_for_frame(rows, frame_id=frame_id),
        lambda: build_pyg_graph_for_frame(rows, frame_id),
    ]
    last = None
    for fn in attempts:
        try:
            return fn()
        except TypeError as e:
            last = e
            continue
    raise RuntimeError(
        f"could not call build_pyg_graph_for_frame; last TypeError: {last}"
    )


def build_graphs(rows: list[dict], graphs_dir: Path, video_id: str) -> int:
    import torch

    graphs_dir.mkdir(parents=True, exist_ok=True)
    by_frame: dict[int, list[dict]] = {}
    for r in rows:
        by_frame.setdefault(int(r["frame_idx"]), []).append(r)

    count = 0
    for frame_id in sorted(by_frame):
        data = _call_builder(by_frame[frame_id], frame_id, video_id)
        try:
            data.video_id = video_id
        except Exception:
            pass
        if not hasattr(data, "frame_id") or data.frame_id is None:
            try:
                data.frame_id = frame_id
            except Exception:
                pass
        out = graphs_dir / f"graph_f{frame_id:06d}.pt"
        torch.save(data, out)
        count += 1
    return count


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/fixtures/demo")
    args = p.parse_args(argv)

    out = Path(args.out)
    out_dir = out if out.is_absolute() else REPO_ROOT / out

    np.random.seed(SEED)

    rows = make_tracks()
    write_tracks(rows, out_dir / "tracks.jsonl")
    print(f"wrote {out_dir / 'tracks.jsonl'} ({len(rows)} rows)")

    write_homography(make_homography(), out_dir / "homography.yaml")
    print(f"wrote {out_dir / 'homography.yaml'}")

    n = build_graphs(rows, out_dir / "graphs", video_id="demo_fixture")
    print(f"wrote {n} graphs to {out_dir / 'graphs'}")

    meta = {
        "seed": SEED,
        "num_frames": NUM_FRAMES,
        "fps": FPS,
        "num_actors": len(ACTORS),
        "video_id": "demo_fixture",
    }
    (out_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {out_dir / 'meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
