import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from graf.graph.builders import (
    GraphBuilder,
    compute_feature_stats,
    build_graph_for_frame,
    build_pyg_graph_for_frame,
    _legacy_group_by_frame,
    _legacy_trim_graph_edges,
)


def _make_actor(track_id=1, actor_class="car", x=0.0, y=0.0, vx=0.0, vy=0.0):
    return {
        "track_id": track_id,
        "actor_class": actor_class,
        "x_m": x,
        "y_m": y,
        "vx": vx,
        "vy": vy,
    }


def test_compute_feature_stats_basic():
    g1 = Data(
        x=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
    )
    g2 = Data(
        x=torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_attr=torch.tensor([[0.5, 0.6], [0.7, 0.8]]),
    )
    stats = compute_feature_stats([g1, g2])
    assert stats.node_mean is not None
    assert stats.node_std is not None
    assert stats.edge_mean is not None
    assert stats.edge_std is not None
    assert np.allclose(stats.node_mean, [4.0, 5.0])
    assert np.allclose(stats.edge_mean, [0.4, 0.5])


def test_compute_feature_stats_empty():
    stats = compute_feature_stats([Data()])
    assert stats.node_mean is None
    assert stats.edge_mean is None


def test_build_graph_for_frame_empty():
    result = build_graph_for_frame([], radius=6.0)
    assert result["num_nodes"] == 0
    assert result["num_edges"] == 0
    assert result["nodes"] == []
    assert result["edges"] == []


def test_build_graph_for_frame_basic():
    records = [
        _make_actor(1, "car", 0.0, 0.0, 1.0, 0.0),
        _make_actor(2, "pedestrian", 1.0, 0.0, 0.5, 0.0),
    ]
    result = build_graph_for_frame(records, radius=10.0)
    assert result["num_nodes"] == 2
    assert result["num_edges"] > 0
    assert all(e["src_node"] != e["dst_node"] for e in result["edges"])
    assert result["nodes"][0]["track_id"] == 1


def test_build_pyg_graph_for_frame_basic():
    actors = [
        _make_actor(1, "car", 0.0, 0.0, 1.0, 0.0),
        _make_actor(2, "pedestrian", 1.0, 0.0, 0.5, 0.0),
    ]
    data = build_pyg_graph_for_frame(
        actors, radius=10.0, frame_id=5, video_id="v1", directed=False
    )
    assert isinstance(data, Data)
    assert data.num_nodes == 2
    assert data.frame_id == 5
    assert data.video_id == "v1"


def test_legacy_group_by_frame():
    records = [
        {"frame_id": 1, "track_id": "a"},
        {"frame_id": 1, "track_id": "b"},
        {"frame_id": 2, "track_id": "a"},
    ]
    grouped = _legacy_group_by_frame(records)
    assert set(grouped.keys()) == {1, 2}
    assert len(grouped[1]) == 2
    assert len(grouped[2]) == 1


def test_legacy_trim_graph_edges_skips_self_loop_and_picks_first():
    graph = {
        "edges": [
            {"source": 1, "target": 1},   # self loop skip
            {"source": 2, "target": 3},   # valid
        ]
    }
    trimmed = _legacy_trim_graph_edges(graph)
    assert len(trimmed["edges"]) == 1
    assert trimmed["edges"][0]["source"] == 2

    # If only self-loops, legacy fallback keeps first edge
    graph2 = {"edges": [{"source": 1, "target": 1}]}
    trimmed2 = _legacy_trim_graph_edges(graph2)
    assert trimmed2["edges"] == [{"source": 1, "target": 1}]
    assert trimmed2["num_edges"] == 1


def test_graph_builder_validate_graph():
    actors = [
        _make_actor(1, "car", 0.0, 0.0),
        _make_actor(2, "pedestrian", 1.0, 0.0),
    ]
    builder = GraphBuilder(radius=10.0)
    data = builder.build_pyg_data(actors, frame_id=1, video_id="v1")
    checks = builder.validate_graph(data)
    assert checks["has_nodes"] is True
    assert checks["has_edges"] is True
    assert checks["finite_node_features"] is True
    assert checks["finite_edge_features"] is True
    assert checks["valid_edge_indices"] is True
    assert checks["no_self_loops"] is True


def test_graph_builder_find_candidate_pairs_bruteforce():
    builder = GraphBuilder(radius=10.0, use_kdtree=False)
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 10.0]])
    pairs = builder._find_candidate_pairs(pos)
    # All pairs brute force, but distance filtering happens later
    assert len(pairs) == 3
    assert (0, 1) in pairs


def test_graph_builder_find_candidate_pairs_self_loops():
    builder = GraphBuilder(include_self_loops=True, use_kdtree=False)
    pos = np.array([[0.0, 0.0], [1.0, 0.0]])
    pairs = builder._find_candidate_pairs(pos)
    assert (0, 0) in pairs
    assert (1, 1) in pairs


def test_graph_builder_normalize_array():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    mean = np.array([2.0, 3.0], dtype=np.float32)
    std = np.array([1.0, 1.0], dtype=np.float32)
    out = GraphBuilder._normalize_array(arr, mean, std)
    assert np.allclose(out, [[-1.0, -1.0], [1.0, 1.0]])

    # Zero std should not divide by zero
    std_zero = np.array([0.0, 0.0], dtype=np.float32)
    out2 = GraphBuilder._normalize_array(arr, mean, std_zero)
    assert np.allclose(out2, arr - mean)


def test_graph_builder_build_pyg_data_empty_actors():
    builder = GraphBuilder(radius=6.0)
    data = builder.build_pyg_data([])
    assert data.num_nodes == 0
    assert data.edge_index.shape[1] == 0
    assert data.edge_attr.shape[0] == 0
