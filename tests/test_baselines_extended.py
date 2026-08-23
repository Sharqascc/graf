import numpy as np
import pytest
from torch_geometric.data import Data

from graf.models.baselines import (
    GraphFeatureExtractor,
    LogisticRegressionBaseline,
    MajorityClassBaseline,
    MLPBaseline,
    RandomForestBaseline,
    _safe_stats,
    _to_numpy,
    get_baseline,
)


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------
def test_to_numpy_none():
    out = _to_numpy(None)
    assert isinstance(out, np.ndarray)
    assert out.size == 0


def test_to_numpy_list():
    out = _to_numpy([1, 2, 3])
    assert isinstance(out, np.ndarray)
    assert out.shape == (3,)


def test_safe_stats_empty():
    out = _safe_stats(np.array([]))
    assert out == [0.0, 0.0, 0.0, 0.0]


def test_safe_stats_basic():
    out = _safe_stats(np.array([1.0, 2.0, 3.0, 4.0]))
    assert out[0] == pytest.approx(2.5)
    assert out[1] == pytest.approx(1.118, rel=0.01)
    assert out[2] == pytest.approx(1.0)
    assert out[3] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# GraphFeatureExtractor tests
# ---------------------------------------------------------------------------
def test_extractor_tabular_2d():
    X = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = GraphFeatureExtractor.transform(X)
    assert out.shape == (2, 2)
    assert np.allclose(out, X)


def test_extractor_tabular_1d():
    X = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = GraphFeatureExtractor.transform(X)
    assert out.shape == (1, 3)


def test_extractor_tabular_list_of_numbers():
    X = [1.0, 2.0, 3.0]
    out = GraphFeatureExtractor.transform(X)
    assert out.shape == (1, 3)


def test_extractor_unsupported_input():
    with pytest.raises(TypeError, match="Unsupported input type"):
        GraphFeatureExtractor.transform("invalid")


def test_extractor_from_single_graph():
    data = Data(
        x=[[1.0, 2.0], [3.0, 4.0]],
        edge_index=[[0, 1], [1, 0]],
        edge_attr=[[0.1, 0.2]],
        pos=[[0.0, 0.0], [1.0, 1.0]],
    )
    out = GraphFeatureExtractor.transform(data)
    # The feature vector length may vary, but should be 2D with one row
    assert out.ndim == 2
    assert out.shape[0] == 1
    assert out.dtype == np.float32


def test_extractor_is_tabular():
    assert GraphFeatureExtractor._is_tabular(np.array([1, 2])) is True
    assert GraphFeatureExtractor._is_tabular([1.0, 2.0]) is True
    assert GraphFeatureExtractor._is_tabular("abc") is False


# ---------------------------------------------------------------------------
# MajorityClassBaseline tests
# ---------------------------------------------------------------------------
def test_majority_baseline_fit_predict():
    X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=np.float32)
    y = np.array([1, 1, 0, 1], dtype=np.int64)
    model = MajorityClassBaseline()
    model.fit(X, y)
    assert model.majority_class_ == 1
    preds = model.predict(X)
    assert np.all(preds == 1)
    proba = model.predict_proba(X)
    assert proba.shape == (4, 2)
    assert np.allclose(proba[:, 1], 1.0)


def test_majority_baseline_not_fitted():
    model = MajorityClassBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.array([[0.0]]))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# LogisticRegressionBaseline tests
# ---------------------------------------------------------------------------
def test_logistic_baseline_fit_predict():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = LogisticRegressionBaseline(max_iter=500)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (6,)
    proba = model.predict_proba(X)
    assert proba.shape == (6, 2)


def test_logistic_baseline_not_fitted():
    model = LogisticRegressionBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.array([[0.0]]))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# RandomForestBaseline tests
# ---------------------------------------------------------------------------
def test_random_forest_baseline_fit_predict():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = RandomForestBaseline(n_estimators=5, max_depth=2, random_state=42)
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (6,)
    proba = model.predict_proba(X)
    assert proba.shape == (6, 2)


def test_random_forest_not_fitted():
    model = RandomForestBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.array([[0.0]]))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# MLPBaseline tests
# ---------------------------------------------------------------------------
def test_mlp_baseline_fit_predict():
    X = np.array([[0.0], [0.5], [1.0], [1.5], [2.0], [2.5]], dtype=np.float32)
    y = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    model = MLPBaseline(
        hidden_layer_sizes=(4,),
        max_iter=50,
        early_stopping=False,
        random_state=42,
    )
    model.fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (6,)
    proba = model.predict_proba(X)
    assert proba.shape == (6, 2)


def test_mlp_not_fitted():
    model = MLPBaseline()
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict(np.array([[0.0]]))
    with pytest.raises(RuntimeError, match="fitted"):
        model.predict_proba(np.array([[0.0]]))


# ---------------------------------------------------------------------------
# get_baseline tests
# ---------------------------------------------------------------------------
def test_get_baseline_known():
    model = get_baseline("majority")
    assert isinstance(model, MajorityClassBaseline)
    model = get_baseline("logistic", max_iter=200)
    assert isinstance(model, LogisticRegressionBaseline)
    model = get_baseline("random_forest", n_estimators=5)
    assert isinstance(model, RandomForestBaseline)
    model = get_baseline("mlp", hidden_layer_sizes=(4,))
    assert isinstance(model, MLPBaseline)


def test_get_baseline_unknown():
    with pytest.raises(ValueError, match="Unknown baseline"):
        get_baseline("unknown")
