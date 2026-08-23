import pytest
import torch
from torch_geometric.data import Data
from unittest.mock import patch, MagicMock

from graf.graph.pyg_export import (
    package_tensor_graph,
    save_graph_sample,
    load_graph_sample,
    to_pyg_data,
    to_pyg_dict,
)


def test_package_tensor_graph_basic():
    x = torch.rand(4, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # shape [2, 2]
    edge_attr = torch.rand(2, 3)
    pos = torch.rand(4, 2)
    data = package_tensor_graph(x, edge_index, edge_attr, pos=pos)
    assert isinstance(data, Data)
    assert data.num_nodes == 4
    assert data.edge_index.shape == (2, 2)
    assert data.edge_attr.shape == (2, 3)
    assert data.pos.shape == (4, 2)


def test_package_tensor_graph_infers_pos_from_x():
    x = torch.rand(4, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, 3)
    data = package_tensor_graph(x, edge_index, edge_attr)
    assert torch.allclose(data.pos, x[:, :2])


def test_package_tensor_graph_with_metadata():
    x = torch.rand(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, 3)
    metadata = {"video_id": "vid1", "frame_id": 10}
    data = package_tensor_graph(x, edge_index, edge_attr, metadata=metadata)
    assert data.video_id == "vid1"
    assert data.frame_id == 10


def test_package_tensor_graph_bad_x_dim():
    x = torch.rand(3)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, 3)
    with pytest.raises(ValueError, match="x must be 2D"):
        package_tensor_graph(x, edge_index, edge_attr)


def test_package_tensor_graph_bad_edge_index_shape():
    x = torch.rand(3, 5)
    edge_index = torch.tensor([0, 1])  # 1D
    edge_attr = torch.rand(1, 3)
    with pytest.raises(ValueError, match="edge_index must be"):
        package_tensor_graph(x, edge_index, edge_attr)


def test_package_tensor_graph_edge_attr_mismatch():
    x = torch.rand(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)  # 2 edges
    edge_attr = torch.rand(1, 3)  # only 1 row
    with pytest.raises(ValueError, match="edge_attr rows"):
        package_tensor_graph(x, edge_index, edge_attr)


def test_package_tensor_graph_pos_invalid_dim():
    x = torch.rand(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, 3)
    pos = torch.rand(3, 3)  # not [N,2]
    with pytest.raises(ValueError, match="pos must be"):
        package_tensor_graph(x, edge_index, edge_attr, pos=pos)


def test_save_load_roundtrip(tmp_path):
    x = torch.rand(3, 5)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_attr = torch.rand(2, 3)
    pos = torch.rand(3, 2)
    data = package_tensor_graph(x, edge_index, edge_attr, pos=pos, metadata={"video_id": "vid1", "frame_id": 2})
    path = save_graph_sample(data, tmp_path, prefix="test")
    assert path.exists()
    loaded = load_graph_sample(path)
    assert isinstance(loaded, Data)
    assert loaded.num_nodes == 3
    assert loaded.edge_index.shape == (2, 2)
    assert loaded.video_id == "vid1"
    assert loaded.frame_id == 2


def test_load_graph_sample_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_graph_sample(tmp_path / "nonexistent.pt")


def test_to_pyg_dict_empty():
    graph = {"nodes": [], "edges": []}
    payload = to_pyg_dict(graph)
    assert payload["num_nodes"] == 0
    assert payload["edge_index"] == [[], []]
    assert payload["edge_attr"] == []
    assert payload["track_ids"] == []
    assert payload["x"] == []


def test_to_pyg_dict_with_nodes_and_edges():
    graph = {
        "frame_id": 5,
        "nodes": [
            {"track_id": 1, "features": [1.0, 2.0, 3.0]},
            {"track_id": 2, "features": [4.0, 5.0, 6.0]},
        ],
        "edges": [
            {
                "src_node": 1,
                "dst_node": 2,
                "dx": 0.1,
                "dy": 0.2,
                "distance": 0.3,
                "dvx": 0.4,
                "dvy": 0.5,
                "relative_speed": 0.6,
                "closing_speed": 0.7,
                "ttc": 0.8,
                "bearing_cos": 0.9,
                "bearing_sin": 1.0,
                "rel_heading_cos": 1.1,
                "rel_heading_sin": 1.2,
            }
        ],
    }
    payload = to_pyg_dict(graph)
    assert payload["num_nodes"] == 2
    assert payload["frame_id"] == 5
    assert payload["track_ids"] == [1, 2]
    assert len(payload["x"]) == 2
    assert payload["edge_index"] == [[1], [2]]
    assert len(payload["edge_attr"]) == 1
    assert payload["edge_attr"][0][:2] == [0.1, 0.2]


def test_to_pyg_dict_skips_invalid_edges():
    graph = {
        "nodes": [{"track_id": 1, "features": [1.0]}, {"track_id": 2, "features": [2.0]}],
        "edges": [
            {"src_node": 1, "dst_node": 1},
            {"src_node": 1, "dst_node": 2, "distance": 1.0},
        ],
    }
    payload = to_pyg_dict(graph)
    assert payload["edge_index"] == [[1], [2]]


def test_to_pyg_data_deprecation_warning_and_return():
    with patch("graf.graph.pyg_export.GraphBuilder") as MockGraphBuilder:
        mock_builder = MagicMock()
        MockGraphBuilder.return_value = mock_builder
        mock_data = Data(
            x=torch.rand(2, 5),
            edge_index=torch.tensor([[0, 1]], dtype=torch.long),
            edge_attr=torch.rand(1, 3),
        )
        mock_builder.build_pyg_data.return_value = mock_data

        graph = {
            "frame_id": 1,
            "video_id": "vid1",
            "nodes": [{"track_id": 1, "x": 0.0, "y": 0.0, "actor_class": "car"}],
            "edges": [],
            "label": 1,
            "site_id": "siteA",
        }

        with pytest.warns(DeprecationWarning):
            data = to_pyg_data(graph)

        assert isinstance(data, Data)
        # frame_id/video_id are passed to builder, not necessarily stored
        assert data.y is not None
        assert torch.allclose(data.y, torch.tensor([1.0]))
        assert data.site_id == "siteA"

        mock_builder.build_pyg_data.assert_called_once()
        args, kwargs = mock_builder.build_pyg_data.call_args
        assert kwargs["frame_id"] == 1
        assert kwargs["video_id"] == "vid1"
        assert kwargs["actors"][0]["track_id"] == 1
        assert kwargs["actors"][0]["actor_class"] == "car"
