from __future__ import annotations

import json
from pathlib import Path

from graf.graph.builders import build_graph_for_frame
from graf.graph.pyg_export import to_pyg_dict


def export_graph_samples(out_dir: str | Path) -> Path:
    sample_records = [
        {
            "track_id": 1,
            "frame_id": 1,
            "actor_class": "car",
            "x": 0.0,
            "y": 0.0,
            "vx": 2.0,
            "vy": 0.0,
        },
        {
            "track_id": 2,
            "frame_id": 1,
            "actor_class": "two_wheeler",
            "x": 2.0,
            "y": 1.0,
            "vx": 1.5,
            "vy": 0.2,
        },
        {
            "track_id": 3,
            "frame_id": 1,
            "actor_class": "pedestrian",
            "x": 8.0,
            "y": 0.5,
            "vx": 0.2,
            "vy": 0.0,
        },
    ]
    graph = build_graph_for_frame(
        sample_records, radius=5.0, actor_classes=["car", "two_wheeler", "pedestrian"]
    )
    payload = to_pyg_dict(graph)

    out_path = Path(out_dir) / "sample_graph_pyg.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
