"""Generate a tiny synthetic dataset for end-to-end pipeline testing."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import yaml
from torch_geometric.data import Data

from graf.graph.edges import (
    ACTOR_CLASSES, CLASS_TO_INDEX, SIZE_PRIORS,
    build_edge_feature, edge_feature_dim, edge_feature_to_list,
)

FPS = 25.0
DT = 1.0 / FPS
SIZES = {
    "car": (2.0, 4.5), "pedestrian": (0.6, 1.7),
    "two_wheeler": (0.8, 2.0), "auto_rickshaw": (1.5, 2.5),
    "bus": (2.5, 10.0), "truck": (2.5, 9.0),
    "bicycle": (0.7, 1.8), "other": (1.5, 2.5),
}


def simulate(num_frames):
    actors = [
        {"id": 101, "cls": "car",        "x": -6.0, "y": 0.0,  "vx":  2.0, "vy": 0.0},
        {"id": 102, "cls": "car",        "x":  6.0, "y": 0.0,  "vx": -2.0, "vy": 0.0},
        {"id": 201, "cls": "pedestrian", "x": 30.0, "y": 30.0, "vx":  0.0, "vy": 0.0},
    ]
    frames = []
    for f in range(num_frames):
        snap = []
        for a in actors:
            x = a["x"] + a["vx"] * f * DT
            y = a["y"] + a["vy"] * f * DT
            heading = math.atan2(a["vy"], a["vx"]) if (a["vx"] or a["vy"]) else 0.0
            snap.append({
                "track_id": a["id"], "actor_class": a["cls"],
                "x": x, "y": y, "x_m": x, "y_m": y,
                "vx": a["vx"], "vy": a["vy"], "heading_rad": heading,
            })
        frames.append(snap)
    return frames


def write_tracks(frames, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for fi, snap in enumerate(frames):
            for a in snap:
                w, h = SIZES[a["actor_class"]]
                bbox = [a["x"] - w / 2, a["y"] - h, a["x"] + w / 2, a["y"]]
                f.write(json.dumps({
                    "frame_idx": fi, "track_id": a["track_id"],
                    "bbox_xyxy": bbox, "confidence": 0.9,
                    "actor_class": a["actor_class"],
                }) + "\n")


def build_graph(frame_id, snap, *, video_id, dist_thresh, ttc_positive):
    n = len(snap)
    x = torch.zeros(n, 11 + len(ACTOR_CLASSES), dtype=torch.float32)
    pos = torch.zeros(n, 2, dtype=torch.float32)
    tids = torch.zeros(n, dtype=torch.long)
    cls_idx = torch.zeros(n, dtype=torch.long)

    for i, a in enumerate(snap):
        speed = math.hypot(a["vx"], a["vy"])
        x[i, 0:2] = torch.tensor([a["x"], a["y"]])
        x[i, 2:4] = torch.tensor([a["vx"], a["vy"]])
        x[i, 4] = speed
        x[i, 8] = math.sin(a["heading_rad"])
        x[i, 9] = math.cos(a["heading_rad"])
        x[i, 10] = SIZE_PRIORS[a["actor_class"]]
        x[i, 11 + CLASS_TO_INDEX[a["actor_class"]]] = 1.0
        pos[i] = torch.tensor([a["x"], a["y"]])
        tids[i] = a["track_id"]
        cls_idx[i] = CLASS_TO_INDEX[a["actor_class"]]

    src, dst, attrs = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            feat = build_edge_feature(snap[i], snap[j])
            if feat["distance"] <= dist_thresh:
                src.append(i)
                dst.append(j)
                attrs.append(edge_feature_to_list(feat))

    edge_index = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
    edge_attr = (torch.tensor(attrs, dtype=torch.float32)
                 if attrs else torch.zeros(0, edge_feature_dim()))
    y_val = 1.0 if any(a[11] < ttc_positive for a in attrs) else 0.0

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos,
        y=torch.tensor([y_val], dtype=torch.float32),
        track_ids=tids, actor_class_index=cls_idx,
        frame_id=frame_id, video_id=video_id,
    )


def generate(root, *, video_id="synthetic", num_frames=100,
             dist_thresh=10.0, ttc_positive=3.0):
    root = Path(root)
    frames = simulate(num_frames)
    tracks_p = root / "tracks" / video_id / "tracks.jsonl"
    graphs_d = root / "graphs" / video_id
    H_p = root / "homography.yaml"

    write_tracks(frames, tracks_p)
    graphs_d.mkdir(parents=True, exist_ok=True)
    for fi, snap in enumerate(frames):
        g = build_graph(fi, snap, video_id=video_id,
                        dist_thresh=dist_thresh, ttc_positive=ttc_positive)
        torch.save(g, graphs_d / f"{video_id}_f{fi:06d}.pt")

    H_p.write_text(yaml.safe_dump(
        {"H": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]}
    ))
    return {"tracks": tracks_p, "graphs": graphs_d, "homography": H_p}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--video_id", default="synthetic")
    ap.add_argument("--num_frames", type=int, default=100)
    args, _ = ap.parse_known_args(argv)
    paths = generate(args.root, video_id=args.video_id, num_frames=args.num_frames)
    for k, v in paths.items():
        print(f"{k:12s} -> {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
