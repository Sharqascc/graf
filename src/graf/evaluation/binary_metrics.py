from __future__ import annotations

from typing import Iterable


def _to_list(values: Iterable) -> list[float]:
    out = []
    for v in values:
        out.append(float(v))
    return out


def sigmoid(x: float) -> float:
    import math
    return 1.0 / (1.0 + math.exp(-x))


def binary_classification_metrics(y_true: Iterable, logits: Iterable, threshold: float = 0.5) -> dict[str, float]:
    y_true = [int(v) for v in y_true]
    logits = _to_list(logits)
    probs = [sigmoid(v) for v in logits]
    y_pred = [1 if p >= threshold else 0 for p in probs]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    total = max(len(y_true), 1)
    accuracy = (tp + tn) / total
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)

    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
        metrics["pr_auc"] = float(average_precision_score(y_true, probs))
    except Exception:
        pass

    return metrics
