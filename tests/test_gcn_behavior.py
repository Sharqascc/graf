"""Behavioral tests for the GCN risk model (item 12).

The current model in gcn_risk.py uses GCNConv, which accepts only
scalar edge weights. The 15-dim edge_attr that carries TTC never
reaches the model. Item 13 landed the edge-aware fix (GINEConv, edge_dim=15,
pool=max). The two tests that were xfail(strict=True) now run
as normal assertions and must stay green. If they regress,
edge features have stopped reaching the model.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric.data")

from torch_geometric.data import Data

from graf.models.gcn_risk import build_model

NODE_DIM = 19
EDGE_DIM = 15


def _small_graph(edge_attr_val: float = 1.0) -> Data:
    """Four nodes, three edges, deterministic features."""
    x = torch.arange(4 * NODE_DIM, dtype=torch.float32).reshape(4, NODE_DIM)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.full((3, EDGE_DIM), edge_attr_val, dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# ---------------------------------------------------------------------
# Two xfail(strict=True) tests: fail today, pass after item 13
# ---------------------------------------------------------------------


def test_output_depends_on_edge_attr():
    """Changing edge_attr must change the model output.

    The smoking gun for the GCNConv bug: the 15 edge features
    carry TTC (column 11), and the label is a function of TTC.
    A model that ignores edge_attr cannot see the label.
    """
    g_a = _small_graph(edge_attr_val=0.0)
    g_b = _small_graph(edge_attr_val=1.0)
    model = build_model(
        in_channels=NODE_DIM, hidden_channels=16, edge_dim=EDGE_DIM
    )
    model.eval()
    with torch.no_grad():
        out_a = model(g_a)
        out_b = model(g_b)
    assert not torch.allclose(out_a, out_b, atol=1e-6)


def test_model_accepts_edge_dim_kwarg():
    """A model that uses edge features must accept edge_dim."""
    model = build_model(in_channels=NODE_DIM, hidden_channels=16, edge_dim=EDGE_DIM)
    assert model is not None


# ---------------------------------------------------------------------
# Six tests that pass today and remain green after the fix
# ---------------------------------------------------------------------


def test_output_shape_single_graph():
    graph = _small_graph()
    model = build_model(in_channels=NODE_DIM, hidden_channels=16)
    model.eval()
    with torch.no_grad():
        out = model(graph)
    assert out.shape == (1,)


def test_output_shape_batched_graph():
    x = torch.zeros((6, NODE_DIM), dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 3, 4], [1, 2, 4, 5]], dtype=torch.long)
    batch = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)
    graph = Data(x=x, edge_index=edge_index, batch=batch)
    model = build_model(in_channels=NODE_DIM, hidden_channels=16)
    model.eval()
    with torch.no_grad():
        out = model(graph)
    assert out.shape == (2,)


def test_determinism_same_seed():
    graph = _small_graph()
    torch.manual_seed(7)
    m1 = build_model(in_channels=NODE_DIM, hidden_channels=16)
    torch.manual_seed(7)
    m2 = build_model(in_channels=NODE_DIM, hidden_channels=16)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1 = m1(graph)
        out2 = m2(graph)
    assert torch.allclose(out1, out2, atol=1e-6)


def test_gradients_reach_every_parameter():
    graph = _small_graph()
    model = build_model(in_channels=NODE_DIM, hidden_channels=16)
    model.train()
    out = model(graph)
    out.sum().backward()
    for name, param in model.named_parameters():
        assert param.grad is not None, "no gradient for " + name


def test_pool_variants_produce_different_outputs():
    graph = _small_graph()
    torch.manual_seed(0)
    m_mean = build_model(in_channels=NODE_DIM, hidden_channels=16, pool="mean")
    torch.manual_seed(0)
    m_max = build_model(in_channels=NODE_DIM, hidden_channels=16, pool="max")
    m_mean.eval()
    m_max.eval()
    with torch.no_grad():
        out_mean = m_mean(graph)
        out_max = m_max(graph)
    assert not torch.allclose(out_mean, out_max, atol=1e-6)


def test_model_state_dict_roundtrip():
    graph = _small_graph()
    torch.manual_seed(42)
    m1 = build_model(in_channels=NODE_DIM, hidden_channels=16)
    state = m1.state_dict()
    torch.manual_seed(99)
    m2 = build_model(in_channels=NODE_DIM, hidden_channels=16)
    m2.load_state_dict(state)
    m1.eval()
    m2.eval()
    with torch.no_grad():
        out1 = m1(graph)
        out2 = m2(graph)
    assert torch.allclose(out1, out2, atol=1e-6)
