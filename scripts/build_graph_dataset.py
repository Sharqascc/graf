from __future__ import annotations

import json
from pathlib import Path

from graf.data.graph_dataset import GraphSampleDataset
from graf.graph.builders import build_graph_for_frame


SAMPLES = [
    {
        "graph": build_graph_for_frame([
            {"track_id": 1, "frame_id": 10, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
            {"track_id": 2, "frame_id": 10, "actor_class": "two_wheeler", "x": 2.0, "y": 1.0, "vx": 1.2, "vy": 0.1},
        ], radius=5.0, actor_classes=["car", "two_wheeler"]),
        "label": 1,
    },
    {
        "graph": build_graph_for_frame([
            {"track_id": 3, "frame_id": 11, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
            {"track_id": 4, "frame_id": 11, "actor_class": "pedestrian", "x": 9.0, "y": 1.0, "vx": 0.3, "vy": 0.0},
        ], radius=5.0, actor_classes=["car", "pedestrian"]),
        "label": 0,
    },
]


def main() -> None:
    out_dir = Path("outputs/datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = out_dir / "graph_samples.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in SAMPLES:
            f.write(json.dumps(row) + "\n")

    dataset = GraphSampleDataset.from_jsonl(jsonl_path)
    print(f"Built dataset with {len(dataset)} samples at {jsonl_path}")


if __name__ == "__main__":
    main()
