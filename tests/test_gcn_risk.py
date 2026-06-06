import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

from torch_geometric.data import Data
from graf.models.gcn_risk import build_model, has_torch_geometric


def test_has_torch_geometric():
    assert has_torch_geometric() is True


def test_build_model_returns_real_model():
    model = build_model(in_channels=6, hidden_channels=16)
    assert model.__class__.__name__ == "GCNRiskModel"


def test_build_model_forward_single_graph():
    model = build_model(in_channels=6, hidden_channels=16, num_layers=2, dropout=0.1, pool="mean")
    x = torch.randn((4, 6))
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index)

    out = model(data)
    assert out.shape == (1,)


def test_build_model_forward_batched_graphs():
    model = build_model(in_channels=6, hidden_channels=16, num_layers=3, dropout=0.1, pool="add")
    x = torch.randn((5, 6))
    edge_index = torch.tensor([[0, 1, 3], [1, 2, 4]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    out = model(data)
    assert out.shape == (2,)


def test_invalid_num_layers_raises():
    with pytest.raises(ValueError):
        build_model(in_channels=6, num_layers=1)


def test_invalid_pool_raises():
    with pytest.raises(ValueError):
        build_model(in_channels=6, pool="median")
