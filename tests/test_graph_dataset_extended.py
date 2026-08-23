from pathlib import Path

import pytest
import torch
from torch_geometric.data import Data

from graf.data.dataset import PtGraphDataset
from graf.data.graph_dataset import (
    LegacyGraphSampleDataset,
    SpatioTemporalWindowDataset,
)


# --- LegacyGraphSampleDataset ---
def test_legacy_dataset_normalize_basic():
    row = {
        "graph": {"nodes": [{"track_id": "a"}], "edges": [], "label": 1, "frame_id": 5},
        "sample_id": "s1",
    }
    ds = LegacyGraphSampleDataset([row])
    assert len(ds) == 1
    sample = ds[0]
    assert sample["nodes"] == [{"track_id": "a"}]
    assert sample["edges"] == []
    assert sample["label"] == 1
    assert sample["sample_id"] == "s1"
    assert sample["frame_id"] == 5
    assert sample["y"] == [1.0]


def test_legacy_dataset_normalize_missing_graph():
    row = {"label": 0, "sample_id": "s2"}
    ds = LegacyGraphSampleDataset([row])
    sample = ds[0]
    assert sample["nodes"] == []
    assert sample["edges"] == []
    assert sample["label"] == 0
    assert sample["sample_id"] == "s2"


def test_legacy_dataset_normalize_graph_not_dict():
    row = {"graph": "invalid", "sample_id": "s3"}
    ds = LegacyGraphSampleDataset([row])
    sample = ds[0]
    assert sample["nodes"] == []
    assert sample["label"] == 0


def test_legacy_dataset_normalize_frame_id_fallback():
    row = {"graph": {"nodes": [], "frame_idx": 10}}
    ds = LegacyGraphSampleDataset([row])
    assert ds[0]["frame_id"] == 10


def test_legacy_dataset_normalize_track_ids_fallback():
    row = {
        "graph": {
            "nodes": [{"track_id": "a"}, {"track_id": "b"}, {"other": 1}],
            "edges": [],
        }
    }
    ds = LegacyGraphSampleDataset([row])
    assert ds[0]["track_ids"] == ["a", "b"]


def test_legacy_dataset_copies_extra_keys():
    row = {"graph": {"nodes": [], "extra_key": "extra"}, "row_extra": "row"}
    ds = LegacyGraphSampleDataset([row])
    sample = ds[0]
    assert sample["extra_key"] == "extra"
    assert sample["row_extra"] == "row"


def test_legacy_dataset_from_jsonl(tmp_path):
    file = tmp_path / "samples.jsonl"
    file.write_text(
        "\n".join(
            [
                '{"graph": {"nodes": [], "label": 1}, "sample_id": "a"}',
                "",
                '{"graph": {"nodes": [], "label": 0}, "sample_id": "b"}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ds = LegacyGraphSampleDataset.from_jsonl(file)
    assert len(ds) == 2
    assert ds[0]["sample_id"] == "a"
    assert ds[1]["sample_id"] == "b"


# --- PtGraphDataset ---
def _make_graph_pt(path, video_id, frame_id, num_nodes=2):
    data = Data(
        x=torch.rand(num_nodes, 5),
        pos=torch.rand(num_nodes, 2),
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_attr=torch.empty((0, 3)),
        track_ids=torch.arange(num_nodes, dtype=torch.long),
        video_id=video_id,
        frame_id=frame_id,
    )
    torch.save(data, path)
    return path


def test_pt_graph_dataset_basic(tmp_path):
    _make_graph_pt(tmp_path / "f0.pt", "v1", 0)
    _make_graph_pt(tmp_path / "f1.pt", "v1", 1)
    ds = PtGraphDataset(tmp_path)
    assert len(ds) == 2
    data = ds[0]
    assert isinstance(data, Data)
    assert data.video_id == "v1"
    assert data.frame_id in (0, 1)
    assert isinstance(ds.file_path(0), Path)


def test_pt_graph_dataset_missing_dir(tmp_path):
    with pytest.raises(FileNotFoundError):
        PtGraphDataset(tmp_path / "missing")


def test_pt_graph_dataset_recursive(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    _make_graph_pt(tmp_path / "root.pt", "v1", 0)
    _make_graph_pt(sub / "nested.pt", "v1", 1)
    ds = PtGraphDataset(tmp_path, recursive=True)
    assert len(ds) == 2
    ds_nonrec = PtGraphDataset(tmp_path, recursive=False)
    assert len(ds_nonrec) == 1


# --- SpatioTemporalWindowDataset ---
def test_window_dataset_basic(tmp_path):
    for i in range(4):
        _make_graph_pt(tmp_path / f"v1_{i}.pt", "v1", i, num_nodes=2)
    for i in range(2):
        _make_graph_pt(tmp_path / f"v2_{i}.pt", "v2", i, num_nodes=2)

    ds = SpatioTemporalWindowDataset(tmp_path, window_size=3, stride=1)
    assert len(ds) == 2

    item = ds[0]
    assert isinstance(item, Data)
    assert item.window_size == 3
    assert item.window_index == 0


def test_window_dataset_stride(tmp_path):
    for i in range(5):
        _make_graph_pt(tmp_path / f"f{i}.pt", "v1", i, num_nodes=2)
    ds = SpatioTemporalWindowDataset(tmp_path, window_size=3, stride=2)
    assert len(ds) == 2


def test_window_dataset_invalid_params(tmp_path):
    with pytest.raises(ValueError, match="window_size must be positive"):
        SpatioTemporalWindowDataset(tmp_path, window_size=0)
    with pytest.raises(ValueError, match="stride must be positive"):
        SpatioTemporalWindowDataset(tmp_path, stride=0)
