import pytest
import torch
from torch_geometric.data import Data

from graf.graph.edges import EDGE_FEATURE_ORDER
from graf.graph.features import (
    extract_class_one_hot,
    extract_edge_angles,
    extract_edge_distance,
    extract_edge_kinematics,
    extract_edge_risk_features,
    extract_heading_features,
    extract_kinematic_features,
    extract_node_base_features,
    num_edge_features,
    num_node_base_features,
)


def test_num_node_base_features():
    assert num_node_base_features() == 11

def test_num_edge_features():
    assert num_edge_features() == len(EDGE_FEATURE_ORDER)

def test_extract_node_base_features_valid():
    x = torch.rand(4, 15)
    data = Data(x=x)
    out = extract_node_base_features(data)
    assert out.shape == (4, 11)
    assert torch.allclose(out, x[:, :11])

def test_extract_node_base_features_missing_x():
    data = Data(pos=torch.rand(3,2))
    out = extract_node_base_features(data)
    assert out.shape == (0, 11)

def test_extract_node_base_features_empty():
    data = Data(x=torch.empty(0, 15))
    out = extract_node_base_features(data)
    assert out.shape == (0, 11)

def test_extract_node_base_features_insufficient_dim():
    data = Data(x=torch.rand(2, 5))
    with pytest.raises(ValueError, match="at least 11"):
        extract_node_base_features(data)

def test_extract_kinematic_features_valid():
    x = torch.rand(4, 15)
    data = Data(x=x)
    out = extract_kinematic_features(data)
    assert out.shape == (4, 8)
    assert torch.allclose(out, x[:, :8])

def test_extract_kinematic_features_missing_x():
    data = Data(pos=torch.rand(3,2))
    out = extract_kinematic_features(data)
    assert out.shape == (0, 8)

def test_extract_kinematic_features_insufficient_dim():
    data = Data(x=torch.rand(2, 5))
    with pytest.raises(ValueError, match="at least 8"):
        extract_kinematic_features(data)

def test_extract_heading_features_valid():
    x = torch.rand(4, 15)
    data = Data(x=x)
    out = extract_heading_features(data)
    assert out.shape == (4, 2)
    assert torch.allclose(out, x[:, 8:10])

def test_extract_heading_features_missing_x():
    data = Data(pos=torch.rand(3,2))
    out = extract_heading_features(data)
    assert out.shape == (0, 2)

def test_extract_heading_features_insufficient_dim():
    data = Data(x=torch.rand(2, 5))
    with pytest.raises(ValueError, match="at least 10"):
        extract_heading_features(data)

def test_extract_class_one_hot_valid():
    x = torch.rand(4, 15)
    data = Data(x=x)
    out = extract_class_one_hot(data)
    assert out.shape == (4, 4)
    assert torch.allclose(out, x[:, 11:15])

def test_extract_class_one_hot_missing_x():
    data = Data(pos=torch.rand(3,2))
    out = extract_class_one_hot(data)
    assert out.shape == (0, 0)

def test_extract_class_one_hot_empty():
    data = Data(x=torch.empty(0, 15))
    out = extract_class_one_hot(data)
    assert out.shape == (0, 4)

def test_extract_class_one_hot_no_class_columns():
    data = Data(x=torch.rand(2, 11))
    out = extract_class_one_hot(data)
    assert out.shape == (2, 0)

def test_extract_edge_distance_valid():
    edge_attr = torch.rand(5, len(EDGE_FEATURE_ORDER))
    data = Data(edge_attr=edge_attr)
    out = extract_edge_distance(data)
    assert out.shape == (5, 1)
    assert torch.allclose(out, edge_attr[:, 2:3])

def test_extract_edge_distance_missing_edge_attr():
    data = Data(x=torch.rand(3, 11))
    out = extract_edge_distance(data)
    assert out.shape == (0, 1)

def test_extract_edge_distance_empty():
    data = Data(edge_attr=torch.empty(0, len(EDGE_FEATURE_ORDER)))
    out = extract_edge_distance(data)
    assert out.shape == (0, 1)

def test_extract_edge_distance_insufficient_dim():
    data = Data(edge_attr=torch.rand(2, 2))
    with pytest.raises(ValueError, match="at least 3"):
        extract_edge_distance(data)

def test_extract_edge_kinematics_valid():
    edge_attr = torch.rand(5, len(EDGE_FEATURE_ORDER))
    data = Data(edge_attr=edge_attr)
    out = extract_edge_kinematics(data)
    expected_idx = [0, 1, 2, 3, 4, 5, 6, 11]
    assert out.shape == (5, 8)
    assert torch.allclose(out, edge_attr[:, expected_idx])

def test_extract_edge_kinematics_missing_edge_attr():
    data = Data(x=torch.rand(3, 11))
    out = extract_edge_kinematics(data)
    assert out.shape == (0, 8)

def test_extract_edge_kinematics_insufficient_dim():
    data = Data(edge_attr=torch.rand(2, 5))
    with pytest.raises(ValueError, match=f"Expected {len(EDGE_FEATURE_ORDER)}"):
        extract_edge_kinematics(data)

def test_extract_edge_angles_valid():
    edge_attr = torch.rand(5, len(EDGE_FEATURE_ORDER))
    data = Data(edge_attr=edge_attr)
    out = extract_edge_angles(data)
    expected_idx = [7, 8, 9, 10]
    assert out.shape == (5, 4)
    assert torch.allclose(out, edge_attr[:, expected_idx])

def test_extract_edge_angles_missing_edge_attr():
    data = Data(x=torch.rand(3, 11))
    out = extract_edge_angles(data)
    assert out.shape == (0, 4)

def test_extract_edge_angles_insufficient_dim():
    data = Data(edge_attr=torch.rand(2, 5))
    with pytest.raises(ValueError, match=f"Expected {len(EDGE_FEATURE_ORDER)}"):
        extract_edge_angles(data)

def test_extract_edge_risk_features_valid():
    edge_attr = torch.rand(5, len(EDGE_FEATURE_ORDER))
    data = Data(edge_attr=edge_attr)
    out = extract_edge_risk_features(data)
    expected_idx = [2, 5, 6, 11, 12, 13, 14]
    assert out.shape == (5, 7)
    assert torch.allclose(out, edge_attr[:, expected_idx])

def test_extract_edge_risk_features_missing_edge_attr():
    data = Data(x=torch.rand(3, 11))
    out = extract_edge_risk_features(data)
    assert out.shape == (0, 7)

def test_extract_edge_risk_features_insufficient_dim():
    data = Data(edge_attr=torch.rand(2, 5))
    with pytest.raises(ValueError, match=f"Expected {len(EDGE_FEATURE_ORDER)}"):
        extract_edge_risk_features(data)
