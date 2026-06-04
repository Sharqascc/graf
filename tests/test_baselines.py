import numpy as np
import pytest

torch = pytest.importorskip("torch")
pyg_data = pytest.importorskip("torch_geometric.data")

Data = pyg_data.Data
Batch = pyg_data.Batch

from graf.models.baselines import GraphFeatureExtractor, MajorityClassBaseline, get_baseline


def test_graph_feature_extractor_single_graph():
    x = torch.randn((4, 6))
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_attr = torch.randn((3, 5))
    y = torch.tensor([1])
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

    features = GraphFeatureExtractor.transform(graph)
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[0] == 1
    assert features.shape[1] > 10


def test_graph_feature_extractor_batch():
    g1 = Data(
        x=torch.randn((4, 6)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn((2, 5)),
        y=torch.tensor([0]),
    )
    g2 = Data(
        x=torch.randn((2, 6)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.randn((1, 5)),
        y=torch.tensor([1]),
    )
    batch = Batch.from_data_list([g1, g2])

    features = GraphFeatureExtractor.transform(batch)
    assert features.shape[0] == 2
    assert features.ndim == 2


def test_majority_baseline_predict_proba_sets_classes():
    X = np.array([[0.0], [1.0], [2.0]], dtype=np.float32)
    y = np.array([1, 1, 0], dtype=np.int64)

    model = MajorityClassBaseline().fit(X, y)
    proba = model.predict_proba(X)

    assert hasattr(model, "classes_")
    assert model.classes_ is not None
    assert proba.shape == (3, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_random_forest_baseline_accepts_pyg_batch():
    g1 = Data(
        x=torch.randn((4, 6)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.randn((2, 5)),
        y=torch.tensor([0]),
    )
    g2 = Data(
        x=torch.randn((3, 6)),
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_attr=torch.randn((1, 5)),
        y=torch.tensor([1]),
    )
    batch = Batch.from_data_list([g1, g2])
    y = np.array([0, 1], dtype=np.int64)

    model = get_baseline("random_forest", n_estimators=10, random_state=42)
    model.fit(batch, y)
    preds = model.predict(batch)
    proba = model.predict_proba(batch)

    assert len(preds) == 2
    assert proba.shape[0] == 2
