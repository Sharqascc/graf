import json
from pathlib import Path

from graf.data.graph_dataset import GraphSampleDataset
from graf.graph.builders import build_graph_for_frame


def make_graph(frame_id: int, x2: float):
    records = [
        {"track_id": 1, "frame_id": frame_id, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
        {"track_id": 2, "frame_id": frame_id, "actor_class": "two_wheeler", "x": x2, "y": 1.0, "vx": 1.0, "vy": 0.2},
    ]
    return build_graph_for_frame(records, radius=5.0, actor_classes=["car", "two_wheeler"])


def test_graph_sample_dataset_len_and_item():
    dataset = GraphSampleDataset([
        {"graph": make_graph(1, 2.0), "label": 1},
        {"graph": make_graph(2, 4.0), "label": 0},
    ])
    assert len(dataset) == 2
    item = dataset[0]
    if isinstance(item, dict):
        assert item["frame_id"] == 1
        assert item["y"] == [1.0]
    else:
        assert item.frame_id == 1
        assert float(item.y.item()) == 1.0


def test_graph_sample_dataset_from_jsonl(tmp_path: Path):
    path = tmp_path / "samples.jsonl"
    rows = [
        {"graph": make_graph(3, 2.0), "label": 1},
        {"graph": make_graph(4, 6.5), "label": 0},
    ]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    dataset = GraphSampleDataset.from_jsonl(path)
    assert len(dataset) == 2
    first = dataset[0]
    if isinstance(first, dict):
        assert first["track_ids"] == [1, 2]
    else:
        assert first.track_ids == [1, 2]
