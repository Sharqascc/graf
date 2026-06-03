import torch
from torch_geometric.data import Data

from graf.graph.features import (
    extract_kinematic_features,
    extract_edge_risk_features,
    filter_subgraph_by_class,
    validate_feature_layout,
)


def test_vectorized_subgraph_extraction():
    x = torch.randn((3, 19), dtype=torch.float32)
    edge_index = torch.tensor(
        [[0, 1, 1, 2],
         [1, 0, 2, 1]],
        dtype=torch.long,
    )
    edge_attr = torch.randn((4, 15), dtype=torch.float32)
    pos = torch.randn((3, 2), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos, num_nodes=3)
    data.actor_class_index = torch.tensor([0, 0, 3], dtype=torch.long)

    validate_feature_layout(data)

    kinematics = extract_kinematic_features(data)
    assert kinematics.shape == (3, 8)

    risk = extract_edge_risk_features(data)
    assert risk.shape == (4, 7)

    car_graph = filter_subgraph_by_class(data, target_class_idx=0)
    assert car_graph.num_nodes == 2
    assert car_graph.edge_index.shape[1] == 2
    assert car_graph.edge_attr.shape == (2, 15)


def test_empty_class_subgraph():
    x = torch.randn((2, 19), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_attr = torch.randn((2, 15), dtype=torch.float32)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=2)
    data.actor_class_index = torch.tensor([0, 1], dtype=torch.long)

    sub = filter_subgraph_by_class(data, target_class_idx=3)
    assert sub.num_nodes == 0
    assert sub.edge_index.shape == (2, 0)
    assert sub.edge_attr.shape == (0, 15)
