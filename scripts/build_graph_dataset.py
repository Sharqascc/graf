from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from graf.data.graph_dataset import GraphSampleDataset
from graf.graph.builders import build_graph_for_frame


TOY_SAMPLES = [
    {
        "sample_id": "site01_vid01_win0001",
        "site_id": "site01",
        "video_id": "vid01",
        "window_id": "win0001",
        "label": 1,
        "graph": build_graph_for_frame(
            [
                {"track_id": 1, "frame_id": 10, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
                {"track_id": 2, "frame_id": 10, "actor_class": "two_wheeler", "x": 2.0, "y": 1.0, "vx": 1.2, "vy": 0.1},
            ],
            radius=5.0,
            actor_classes=["car", "two_wheeler"],
        ),
    },
    {
        "sample_id": "site01_vid01_win0002",
        "site_id": "site01",
        "video_id": "vid01",
        "window_id": "win0002",
        "label": 0,
        "graph": build_graph_for_frame(
            [
                {"track_id": 3, "frame_id": 11, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
                {"track_id": 4, "frame_id": 11, "actor_class": "pedestrian", "x": 9.0, "y": 1.0, "vx": 0.3, "vy": 0.0},
            ],
            radius=5.0,
            actor_classes=["car", "pedestrian"],
        ),
    },
    {
        "sample_id": "site02_vid03_win0001",
        "site_id": "site02",
        "video_id": "vid03",
        "window_id": "win0001",
        "label": 1,
        "graph": build_graph_for_frame(
            [
                {"track_id": 5, "frame_id": 20, "actor_class": "auto_rickshaw", "x": 1.0, "y": 1.0, "vx": 1.6, "vy": 0.2},
                {"track_id": 6, "frame_id": 20, "actor_class": "two_wheeler", "x": 3.5, "y": 1.8, "vx": 1.8, "vy": 0.0},
                {"track_id": 7, "frame_id": 20, "actor_class": "pedestrian", "x": 6.0, "y": 2.0, "vx": 0.2, "vy": -0.1},
            ],
            radius=4.5,
            actor_classes=["auto_rickshaw", "two_wheeler", "pedestrian"],
        ),
    },
    {
        "sample_id": "site02_vid04_win0002",
        "site_id": "site02",
        "video_id": "vid04",
        "window_id": "win0002",
        "label": 0,
        "graph": build_graph_for_frame(
            [
                {"track_id": 8, "frame_id": 25, "actor_class": "bus", "x": 0.0, "y": 5.0, "vx": 1.0, "vy": 0.0},
                {"track_id": 9, "frame_id": 25, "actor_class": "car", "x": 8.0, "y": 5.5, "vx": 1.1, "vy": 0.0},
            ],
            radius=3.0,
            actor_classes=["bus", "car"],
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build graph JSONL dataset for GCN risk training.")
    parser.add_argument("--out", type=str, default="data/processed/graphs/site_graphs.jsonl")
    return parser.parse_args()


def validate_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
    required = ["sample_id", "site_id", "video_id", "window_id", "label", "graph"]
    missing = [k for k in required if k not in row]
    if missing:
        raise ValueError(f"Sample {idx} missing keys: {missing}")
    row["label"] = int(row["label"])
    return row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)

    rows = [validate_row(r, i) for i, r in enumerate(TOY_SAMPLES)]
    write_jsonl(out_path, rows)

    dataset = GraphSampleDataset.from_jsonl(out_path)
    positives = sum(int(s.get("label", 0)) for s in dataset.samples)
    negatives = len(dataset) - positives

    print(f"Built dataset with {len(dataset)} samples at {out_path}")
    print(f"Label counts -> positive: {positives}, negative: {negatives}")
    print(f"Sites: {sorted({s.get('site_id', '') for s in dataset.samples})}")


if __name__ == "__main__":
    main()
