import pytest
import torch
from torch_geometric.data import Data

from graf.graph.temporal import build_temporal_window_graph


def make_frame(video_id, frame_id, num_nodes=3, has_y=False):
    x = torch.rand(num_nodes, 5)
    pos = torch.rand(num_nodes, 2)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_attr = torch.rand(4, 3)
    track_ids = torch.arange(num_nodes, dtype=torch.long)
    actor_class_index = torch.randint(0, 3, (num_nodes,), dtype=torch.long)
    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        track_ids=track_ids,
        actor_class_index=actor_class_index,
        video_id=video_id,
        frame_id=frame_id,
    )
    if has_y:
        data.y = torch.rand(1)
    return data


def test_single_frame_no_temporal_edges():
    frames = [make_frame("vid1", 0, num_nodes=3)]
    merged = build_temporal_window_graph(frames)
    assert merged.num_nodes == 3
    assert merged.num_spatial_edges == 4
    assert merged.num_temporal_edges == 0
    assert merged.window_size == 1
    assert merged.video_id == "vid1"
    assert merged.edge_type.tolist() == [0] * 4
    assert merged.num_edges == 4


def test_multiple_frames_with_shared_tracks():
    frames = [make_frame("vid1", 0, num_nodes=3), make_frame("vid1", 1, num_nodes=3)]
    merged = build_temporal_window_graph(frames)
    assert merged.num_nodes == 6
    assert merged.num_spatial_edges == 8
    assert merged.num_temporal_edges == 6
    assert merged.num_edges == 14
    assert merged.window_size == 2
    assert merged.edge_type.tolist() == [0] * 8 + [1] * 6


def test_temporal_self_only_filters_negative_track_ids():
    a = make_frame("vid1", 0, num_nodes=3)
    a.track_ids = torch.tensor([-1, 1, 2], dtype=torch.long)
    b = make_frame("vid1", 1, num_nodes=3)
    b.track_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    merged_self = build_temporal_window_graph([a, b], temporal_self_only=True)
    assert merged_self.num_temporal_edges == 4
    merged_all = build_temporal_window_graph([a, b], temporal_self_only=False)
    assert merged_all.num_temporal_edges == 4


def test_missing_required_attribute_raises():
    bad = Data(
        pos=torch.rand(3, 2),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.rand(1, 2),
        track_ids=torch.arange(3),
    )
    with pytest.raises(ValueError, match="must contain x"):
        build_temporal_window_graph([bad])
    bad2 = Data(
        x=torch.rand(3, 5),
        pos=torch.rand(3, 2),
        edge_attr=torch.rand(1, 2),
        track_ids=torch.arange(3),
    )
    with pytest.raises(ValueError, match="edge_index"):
        build_temporal_window_graph([bad2])
    bad3 = Data(
        x=torch.rand(3, 5),
        pos=torch.rand(3, 2),
        edge_index=torch.tensor([[0], [1]]),
        track_ids=torch.arange(3),
    )
    with pytest.raises(ValueError, match="edge_attr"):
        build_temporal_window_graph([bad3])
    bad4 = Data(
        x=torch.rand(3, 5),
        pos=torch.rand(3, 2),
        edge_index=torch.tensor([[0], [1]]),
        edge_attr=torch.rand(1, 2),
    )
    with pytest.raises(ValueError, match="track_ids"):
        build_temporal_window_graph([bad4])


def test_multiple_video_ids_raises():
    frames = [make_frame("vid1", 0, num_nodes=2), make_frame("vid2", 1, num_nodes=2)]
    with pytest.raises(ValueError, match="same video_id"):
        build_temporal_window_graph(frames)


def test_empty_frames_raises():
    with pytest.raises(ValueError, match="At least one"):
        build_temporal_window_graph([])


def test_temporal_edge_attr_dim_matches_spatial():
    frames = [make_frame("vid1", 0, num_nodes=3), make_frame("vid1", 1, num_nodes=3)]
    merged = build_temporal_window_graph(frames)
    assert merged.edge_attr.shape == (14, 3)
