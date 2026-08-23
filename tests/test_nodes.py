import pytest
import torch
from torch_geometric.data import Data

from graf.graph.nodes import (
    ACTOR_CLASSES,
    BASE_NODE_FEATURES,
    node_feature_dim,
    validate_node_layout,
)


def test_node_feature_dim_default():
    expected = len(BASE_NODE_FEATURES) + len(ACTOR_CLASSES)
    assert node_feature_dim() == expected


def test_node_feature_dim_custom_classes():
    custom = ["car", "pedestrian", "other"]
    expected = len(BASE_NODE_FEATURES) + len(custom)
    assert node_feature_dim(custom) == expected


def test_validate_node_layout_ok():
    num_nodes = 3
    dim = node_feature_dim()
    data = Data(x=torch.rand(num_nodes, dim), pos=torch.rand(num_nodes, 2))
    validate_node_layout(data, require_pos=True)


def test_validate_node_layout_missing_x():
    data = Data(pos=torch.rand(2, 2))
    with pytest.raises(ValueError, match="must contain node feature matrix"):
        validate_node_layout(data)


def test_validate_node_layout_wrong_dim():
    data = Data(x=torch.rand(2, 5), pos=torch.rand(2, 2))
    with pytest.raises(ValueError, match="Unexpected node feature dimension"):
        validate_node_layout(data)


def test_validate_node_layout_pos_mismatch():
    data = Data(x=torch.rand(3, node_feature_dim()), pos=torch.rand(2, 2))
    with pytest.raises(ValueError, match="Mismatch between x and pos"):
        validate_node_layout(data, require_pos=True)
