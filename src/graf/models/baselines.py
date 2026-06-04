"""Baseline models and graph-to-tabular adapters for GRAF."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    import torch
except Exception:  # pragma: no cover
    torch = None

try:
    from torch_geometric.data import Batch, Data
except Exception:  # pragma: no cover
    Batch = None
    Data = None


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([])
    if torch is not None and isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _safe_stats(arr: np.ndarray) -> list[float]:
    arr = np.asarray(arr, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.min(arr)),
        float(np.max(arr)),
    ]


class GraphFeatureExtractor:
    """Convert PyG graph objects into fixed-width tabular feature vectors."""

    @classmethod
    def transform(cls, data: Any) -> np.ndarray:
        if cls._is_tabular(data):
            arr = np.asarray(data, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr

        graphs = cls._normalize_graphs(data)
        features = [cls._graph_to_vector(g) for g in graphs]
        if not features:
            return np.zeros((0, 0), dtype=np.float32)
        return np.asarray(features, dtype=np.float32)

    @classmethod
    def _is_tabular(cls, data: Any) -> bool:
        if isinstance(data, np.ndarray):
            return data.ndim in (1, 2)
        if isinstance(data, (list, tuple)) and data:
            first = data[0]
            if isinstance(first, (int, float, np.number, list, tuple, np.ndarray)):
                return True
        return False

    @classmethod
    def _normalize_graphs(cls, data: Any) -> list[Any]:
        if Batch is not None and isinstance(data, Batch):
            return list(data.to_data_list())
        if Data is not None and isinstance(data, Data):
            return [data]
        if isinstance(data, Sequence):
            return list(data)
        raise TypeError(
            "Unsupported input type for GraphFeatureExtractor.transform(). "
            "Expected ndarray-like, torch_geometric Data/Batch, or list of Data."
        )

    @classmethod
    def _graph_to_vector(cls, graph: Any) -> list[float]:
        x = _to_numpy(getattr(graph, "x", None))
        edge_index = _to_numpy(getattr(graph, "edge_index", None))
        edge_attr = _to_numpy(getattr(graph, "edge_attr", None))
        pos = _to_numpy(getattr(graph, "pos", None))
        y = _to_numpy(getattr(graph, "y", None))

        num_nodes = int(getattr(graph, "num_nodes", x.shape[0] if x.ndim == 2 else 0))
        if edge_index.ndim == 2 and edge_index.shape[0] == 2:
            num_edges = int(edge_index.shape[1])
        elif edge_attr.ndim == 2:
            num_edges = int(edge_attr.shape[0])
        else:
            num_edges = 0

        density = 0.0
        if num_nodes > 1:
            density = float(num_edges / max(1, num_nodes * (num_nodes - 1)))

        node_feat_dim = int(x.shape[1]) if x.ndim == 2 else 0
        edge_feat_dim = int(edge_attr.shape[1]) if edge_attr.ndim == 2 else 0
        pos_feat_dim = int(pos.shape[1]) if pos.ndim == 2 else 0

        vector: list[float] = [
            float(num_nodes),
            float(num_edges),
            float(density),
            float(node_feat_dim),
            float(edge_feat_dim),
            float(pos_feat_dim),
        ]

        if x.ndim == 2 and x.size > 0:
            vector.extend(_safe_stats(x))
            row_l2 = np.linalg.norm(x, axis=1)
            vector.extend(_safe_stats(row_l2))
            vector.append(float(np.count_nonzero(x) / x.size))
        else:
            vector.extend([0.0] * 9)

        if edge_attr.ndim == 2 and edge_attr.size > 0:
            vector.extend(_safe_stats(edge_attr))
            edge_row_l2 = np.linalg.norm(edge_attr, axis=1)
            vector.extend(_safe_stats(edge_row_l2))
            vector.append(float(np.count_nonzero(edge_attr) / edge_attr.size))
        else:
            vector.extend([0.0] * 9)

        if pos.ndim == 2 and pos.size > 0:
            vector.extend(_safe_stats(pos))
        else:
            vector.extend([0.0] * 4)

        if y.size > 0:
            vector.extend(_safe_stats(y.reshape(-1)))
        else:
            vector.extend([0.0] * 4)

        for attr in ["frame_id", "video_id", "track_id", "node_frame_index", "actor_class_index"]:
            arr = _to_numpy(getattr(graph, attr, None))
            if arr.size == 0:
                vector.extend([0.0, 0.0])
            else:
                flat = arr.reshape(-1)
                try:
                    mean_val = float(np.mean(flat.astype(np.float32)))
                except Exception:
                    mean_val = 0.0
                try:
                    uniq_val = float(len(np.unique(flat)))
                except Exception:
                    uniq_val = 0.0
                vector.extend([mean_val, uniq_val])

        return vector


class BaselineClassifierMixin:
    def _prepare_X(self, X: Any) -> np.ndarray:
        arr = GraphFeatureExtractor.transform(X)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D feature matrix, got shape {arr.shape}")
        return arr.astype(np.float32)

    def _prepare_y(self, y: Any) -> np.ndarray:
        return _to_numpy(y).reshape(-1).astype(np.int64)


class MajorityClassBaseline(BaselineClassifierMixin, BaseEstimator, ClassifierMixin):
    def __init__(self) -> None:
        self.majority_class_ = None
        self.classes_ = None
        self.class_prior_ = None

    def fit(self, X: Any, y: Any) -> "MajorityClassBaseline":
        y_arr = self._prepare_y(y)
        self.classes_, counts = np.unique(y_arr, return_counts=True)
        idx = int(np.argmax(counts))
        self.majority_class_ = self.classes_[idx]
        self.class_prior_ = counts / counts.sum()
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self.majority_class_ is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        X_arr = self._prepare_X(X)
        return np.full(X_arr.shape[0], self.majority_class_, dtype=self.classes_.dtype)

    def predict_proba(self, X: Any) -> np.ndarray:
        if self.majority_class_ is None or self.classes_ is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        X_arr = self._prepare_X(X)
        proba = np.zeros((X_arr.shape[0], len(self.classes_)), dtype=np.float32)
        maj_idx = int(np.where(self.classes_ == self.majority_class_)[0][0])
        proba[:, maj_idx] = 1.0
        return proba


class LogisticRegressionBaseline(BaselineClassifierMixin, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        solver: str = "lbfgs",
        class_weight: str | dict | None = "balanced",
        random_state: int | None = 42,
    ) -> None:
        self.C = C
        self.max_iter = max_iter
        self.solver = solver
        self.class_weight = class_weight
        self.random_state = random_state
        self._model = None
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "LogisticRegressionBaseline":
        X_arr = self._prepare_X(X)
        y_arr = self._prepare_y(y)
        self._model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            class_weight=self.class_weight,
            random_state=self.random_state,
        )
        self._model.fit(X_arr, y_arr)
        self.classes_ = getattr(self._model, "classes_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._model.predict(self._prepare_X(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        return self._model.predict_proba(self._prepare_X(X))


class RandomForestBaseline(BaselineClassifierMixin, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        class_weight: str | dict | None = "balanced",
        random_state: int | None = 42,
        n_jobs: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.class_weight = class_weight
        self.random_state = random_state
        self.n_jobs = n_jobs
        self._model = None
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "RandomForestBaseline":
        X_arr = self._prepare_X(X)
        y_arr = self._prepare_y(y)
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self._model.fit(X_arr, y_arr)
        self.classes_ = getattr(self._model, "classes_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._model.predict(self._prepare_X(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        return self._model.predict_proba(self._prepare_X(X))


class MLPBaseline(BaselineClassifierMixin, BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        hidden_layer_sizes=(128, 64, 32),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate: str = "adaptive",
        learning_rate_init: float = 1e-3,
        max_iter: int = 500,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        random_state: int | None = 42,
    ) -> None:
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self._model = None
        self.classes_ = None

    def fit(self, X: Any, y: Any) -> "MLPBaseline":
        X_arr = self._prepare_X(X)
        y_arr = self._prepare_y(y)
        self._model = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            alpha=self.alpha,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            max_iter=self.max_iter,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            n_iter_no_change=self.n_iter_no_change,
            random_state=self.random_state,
        )
        self._model.fit(X_arr, y_arr)
        self.classes_ = getattr(self._model, "classes_", None)
        return self

    def predict(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict().")
        return self._model.predict(self._prepare_X(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model must be fitted before calling predict_proba().")
        return self._model.predict_proba(self._prepare_X(X))


BASELINE_REGISTRY = {
    "majority": MajorityClassBaseline,
    "logistic": LogisticRegressionBaseline,
    "random_forest": RandomForestBaseline,
    "mlp": MLPBaseline,
}


def get_baseline(name: str, **kwargs: Any) -> BaseEstimator:
    if name not in BASELINE_REGISTRY:
        valid = ", ".join(sorted(BASELINE_REGISTRY))
        raise ValueError(f"Unknown baseline {name!r}. Valid options: {valid}")
    return BASELINE_REGISTRY[name](**kwargs)
