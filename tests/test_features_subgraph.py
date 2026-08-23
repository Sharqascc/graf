import pytest
import torch
from torch_geometric.data import Data

from graf.graph.features import (
    filter_subgraph_by_class,
    filter_subgraph_by_node_mask,
    validate_feature_layout,
)


def make_data(num_nodes=5, num_edges=4):
    x = torch.rand(num_nodes, 11)
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 0, 2, 1, 4]], dtype=torch.long)[:, :num_edges]
    edge_attr = torch.rand(num_edges, 15)
    actor_class_index = torch.tensor([0, 1, 1, 0, 1], dtype=torch.long)
    track_ids = torch.arange(num_nodes, dtype=torch.long)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                actor_class_index=actor_class_index, track_ids=track_ids,
                num_nodes=num_nodes)

def test_filter_subgraph_by_class_keep_isolated():
    data = make_data(num_nodes=5, num_edges=4)
    out = filter_subgraph_by_class(data, target_class_idx=1)
    assert out.num_nodes == 3
    assert out.edge_index.shape[1] == 2
    assert out.edge_attr.shape == (2, 15)
    assert out.x.shape == (3, 11)

def test_filter_subgraph_by_class_no_isolated():
    data = make_data(num_nodes=5, num_edges=4)
    out = filter_subgraph_by_class(data, target_class_idx=1, keep_isolated_nodes=False)
    assert out.num_nodes == 2
    assert out.edge_index.shape[1] == 2
    assert out.x.shape == (2, 11)

def test_filter_subgraph_by_class_missing_attribute():
    data = make_data(num_nodes=3, num_edges=1)
    del data.actor_class_index
    with pytest.raises(AttributeError, match="actor_class_index"):
        filter_subgraph_by_class(data, target_class_idx=1)

def test_filter_subgraph_by_class_empty_data():
    data = Data(x=torch.empty(0, 11), edge_index=torch.empty((2,0), dtype=torch.long),
                edge_attr=torch.empty((0,15)), actor_class_index=torch.empty(0, dtype=torch.long),
                num_nodes=0)
    out = filter_subgraph_by_class(data, target_class_idx=1)
    assert out.num_nodes == 0

def test_filter_subgraph_by_node_mask_valid():
    data = make_data(num_nodes=5, num_edges=4)
    mask = torch.tensor([False, True, True, False, True])
    out = filter_subgraph_by_node_mask(data, mask)
    assert out.num_nodes == 3
    assert out.edge_index.shape[1] == 2
    assert out.x.shape == (3, 11)

def test_filter_subgraph_by_node_mask_no_isolated():
    data = make_data(num_nodes=5, num_edges=4)
    mask = torch.tensor([False, True, True, False, True])
    out = filter_subgraph_by_node_mask(data, mask, keep_isolated_nodes=False)
    assert out.num_nodes == 2
    assert out.edge_index.shape[1] == 2
    assert out.x.shape == (2, 11)

def test_filter_subgraph_by_node_mask_bad_dtype():
    data = make_data(num_nodes=3, num_edges=1)
    mask = torch.tensor([1, 0, 1])
    with pytest.raises(TypeError, match="boolean"):
        filter_subgraph_by_node_mask(data, mask)

def test_filter_subgraph_by_node_mask_bad_dim():
    data = make_data(num_nodes=3, num_edges=1)
    mask = torch.ones(3, 1, dtype=torch.bool)
    with pytest.raises(ValueError, match="one-dimensional"):
        filter_subgraph_by_node_mask(data, mask)

def test_filter_subgraph_by_node_mask_wrong_length():
    data = make_data(num_nodes=3, num_edges=1)
    mask = torch.ones(2, dtype=torch.bool)
    with pytest.raises(ValueError, match="does not match num_nodes"):
        filter_subgraph_by_node_mask(data, mask)

def test_filter_subgraph_by_node_mask_empty_subset():
    data = make_data(num_nodes=3, num_edges=1)
    mask = torch.zeros(3, dtype=torch.bool)
    out = filter_subgraph_by_node_mask(data, mask)
    assert out.num_nodes == 0
    assert out.x.shape == (0, 11)
    assert out.edge_attr.shape == (0, 15)

def test_validate_feature_layout_ok():
    data = make_data(num_nodes=3, num_edges=2)
    validate_feature_layout(data)

def test_validate_feature_layout_missing_x():
    data = make_data(num_nodes=3, num_edges=2)
    del data.x
    with pytest.raises(ValueError, match="Data.x is missing"):
        validate_feature_layout(data)

def test_validate_feature_layout_missing_edge_index():
    data = make_data(num_nodes=3, num_edges=2)
    del data.edge_index
    with pytest.raises(ValueError, match="Data.edge_index is missing"):
        validate_feature_layout(data)

def test_validate_feature_layout_missing_edge_attr():
    data = make_data(num_nodes=3, num_edges=2)
    del data.edge_attr
    with pytest.raises(ValueError, match="Data.edge_attr is missing"):
        validate_feature_layout(data)

def test_validate_feature_layout_wrong_x_rank():
    data = make_data(num_nodes=3, num_edges=2)
    data.x = torch.rand(3)
    with pytest.raises(ValueError, match="Data.x must be rank-2"):
        validate_feature_layout(data)

def test_validate_feature_layout_wrong_edge_index_shape():
    data = make_data(num_nodes=3, num_edges=2)
    data.edge_index = torch.rand(2, 2, 2)
    with pytest.raises(ValueError, match="Data.edge_index must have shape"):
        validate_feature_layout(data)

def test_validate_feature_layout_wrong_edge_attr_rows():
    data = make_data(num_nodes=3, num_edges=2)
    data.edge_attr = torch.rand(3, 15)
    with pytest.raises(ValueError, match="edge_attr row count"):
        validate_feature_layout(data)

def test_validate_feature_layout_insufficient_node_features():
    data = make_data(num_nodes=3, num_edges=2)
    data.x = torch.rand(3, 5)
    with pytest.raises(ValueError, match="at least 11"):
        validate_feature_layout(data)

def test_validate_feature_layout_wrong_edge_feature_dim():
    data = make_data(num_nodes=3, num_edges=2)
    data.edge_attr = torch.rand(2, 14)
    with pytest.raises(ValueError, match="Expected 15 edge features"):
        validate_feature_layout(data)
