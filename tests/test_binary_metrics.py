from graf.evaluation.binary_metrics import binary_classification_metrics


def test_binary_metrics_basic_case():
    y_true = [1, 0, 1, 0]
    logits = [2.0, -2.0, 1.5, -1.0]
    metrics = binary_classification_metrics(y_true, logits)
    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_binary_metrics_threshold_effect():
    y_true = [1, 0]
    logits = [0.2, 0.1]
    metrics = binary_classification_metrics(y_true, logits, threshold=0.6)
    assert metrics["accuracy"] == 0.5
