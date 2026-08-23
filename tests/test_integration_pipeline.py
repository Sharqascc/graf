import torch
from torch_geometric.data import Data

from graf.data.dataset import PtGraphDataset
from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.graph.builders import GraphBuilder


def _make_actors(frame_id: int) -> list[dict]:
    return [
        {
            "track_id": 1,
            "frame_id": frame_id,
            "actor_class": "car",
            "x_m": 0.0,
            "y_m": 0.0,
            "vx": 1.0,
            "vy": 0.0,
            "heading_rad": 0.0,
        },
        {
            "track_id": 2,
            "frame_id": frame_id,
            "actor_class": "pedestrian",
            "x_m": 1.5,
            "y_m": 0.0,
            "vx": 0.5,
            "vy": 0.0,
            "heading_rad": 0.0,
        },
    ]


def test_end_to_end_graph_to_temporal_dataset(tmp_path):
    builder = GraphBuilder(radius=5.0)

    # Build two frame graphs
    frames = []
    for frame_id in range(2):
        actors = _make_actors(frame_id)
        data = builder.build_pyg_data(actors, frame_id=frame_id, video_id="v1")
        assert isinstance(data, Data)
        assert data.num_nodes == 2
        assert data.edge_index.shape[1] > 0
        path = tmp_path / f"graph_f{frame_id}.pt"
        torch.save(data, path)
        frames.append(data)

    # Load with PtGraphDataset
    ds = PtGraphDataset(tmp_path)
    assert len(ds) == 2
    loaded = ds[0]
    assert isinstance(loaded, Data)
    assert loaded.video_id == "v1"

    # Build temporal window dataset
    window_ds = SpatioTemporalWindowDataset(tmp_path, window_size=2, stride=1)
    assert len(window_ds) == 1

    window_graph = window_ds[0]
    assert isinstance(window_graph, Data)
    assert window_graph.window_size == 2
    assert window_graph.num_nodes >= 2
    assert window_graph.num_edges > 0
    # Temporal edges should connect same track ids across frames
    assert window_graph.num_temporal_edges > 0
    assert hasattr(window_graph, "track_ids")
    assert hasattr(window_graph, "actor_class_index")
