import pytest
import torch
from torch_geometric.data import Data

from graf.graph.nodes import (
    ACTOR_CLASSES,
    BASE_NODE_FEATURES,
    clone_with_updated_node_feature,
    clone_with_updated_node_feature_block,
    get_actor_class_index,
    get_actor_class_names,
    get_class_one_hot,
    get_heading_components,
    get_kinematic_features,
    get_node_feature_block,
    get_node_feature_index,
    get_node_feature_slice,
    get_position_features,
    get_spatial_positions,
    get_track_ids,
)


def make_data(num_nodes=3, actor_classes=None):
    actor_classes = actor_classes or ACTOR_CLASSES
    base_dim = len(BASE_NODE_FEATURES)
    class_dim = len(actor_classes)
    dim = base_dim + class_dim
    x_base = torch.rand(num_nodes, base_dim)
    class_idx = torch.randint(0, class_dim, (num_nodes,))
    x_class = torch.zeros(num_nodes, class_dim)
    x_class[torch.arange(num_nodes), class_idx] = 1.0
    x = torch.cat([x_base, x_class], dim=1)
    pos = torch.rand(num_nodes, 2)
    track_ids = torch.arange(num_nodes, dtype=torch.long)
    actor_class_index = class_idx.clone()
    return Data(x=x, pos=pos, track_ids=track_ids, actor_class_index=actor_class_index), dim

def test_get_node_feature_index_known():
    assert get_node_feature_index("x") == 0
    assert get_node_feature_index("y") == 1
    assert get_node_feature_index("speed") == 4

def test_get_node_feature_index_unknown():
    with pytest.raises(KeyError):
        get_node_feature_index("nonexistent")

def test_get_node_feature_slice():
    data, dim = make_data(num_nodes=4)
    slice_x = get_node_feature_slice(data, "x")
    assert slice_x.shape == (4,)
    assert torch.allclose(slice_x, data.x[:, 0])

def test_get_node_feature_block():
    data, dim = make_data(num_nodes=5)
    block = get_node_feature_block(data, ["x", "y"])
    assert block.shape == (5, 2)
    assert torch.allclose(block[:, 0], data.x[:, 0])
    assert torch.allclose(block[:, 1], data.x[:, 1])

def test_get_position_features():
    data, dim = make_data(num_nodes=3)
    pos = get_position_features(data)
    assert pos.shape == (3, 2)
    assert torch.allclose(pos, data.x[:, 0:2])

def test_get_kinematic_features():
    data, dim = make_data(num_nodes=3)
    kin = get_kinematic_features(data)
    assert kin.shape == (3, 8)

def test_get_heading_components():
    data, dim = make_data(num_nodes=2)
    heading = get_heading_components(data)
    assert heading.shape == (2, 2)

def test_get_class_one_hot():
    data, dim = make_data(num_nodes=3)
    one_hot = get_class_one_hot(data)
    assert one_hot.shape == (3, len(ACTOR_CLASSES))
    assert torch.allclose(one_hot.sum(dim=1), torch.ones(3))

def test_get_spatial_positions_prefer_pos():
    data, dim = make_data(num_nodes=3)
    pos = get_spatial_positions(data)
    assert torch.allclose(pos, data.pos)

def test_get_spatial_positions_fallback_to_x():
    data, dim = make_data(num_nodes=3)
    data.pos = None
    pos = get_spatial_positions(data)
    assert torch.allclose(pos, data.x[:, 0:2])

def test_get_track_ids():
    data, dim = make_data(num_nodes=3)
    assert torch.allclose(get_track_ids(data), data.track_ids)

def test_get_actor_class_index():
    data, dim = make_data(num_nodes=3)
    assert torch.allclose(get_actor_class_index(data), data.actor_class_index)

def test_get_actor_class_names_from_index():
    data, dim = make_data(num_nodes=3)
    names = get_actor_class_names(data)
    assert len(names) == 3
    assert all(name in ACTOR_CLASSES for name in names)

def test_get_actor_class_names_fallback_to_one_hot():
    data, dim = make_data(num_nodes=3)
    data.actor_class_index = None
    class_idx = 1
    one_hot = torch.zeros(3, len(ACTOR_CLASSES))
    one_hot[:, class_idx] = 1
    data.x = torch.cat([torch.rand(3, len(BASE_NODE_FEATURES)), one_hot], dim=1)
    names = get_actor_class_names(data)
    assert names == [ACTOR_CLASSES[class_idx]] * 3

def test_clone_with_updated_node_feature():
    data, dim = make_data(num_nodes=3)
    new_values = torch.tensor([1.0, 2.0, 3.0])
    updated = clone_with_updated_node_feature(data, "x", new_values)
    assert torch.allclose(updated.x[:, 0], new_values)
    assert not torch.allclose(data.x[:, 0], new_values)

def test_clone_with_updated_node_feature_block():
    data, dim = make_data(num_nodes=3)
    new_values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    updated = clone_with_updated_node_feature_block(data, ["x", "y"], new_values)
    assert torch.allclose(updated.x[:, 0:2], new_values)
    assert not torch.allclose(data.x[:, 0:2], new_values)
