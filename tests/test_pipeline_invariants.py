"""Pipeline invariants that the current suite does not cover.

Each test encodes an invariant the pipeline should satisfy. On the
current commit, the ones marked xfail fail — that is the point: they
document bugs that the existing example-based tests do not catch.
When the corresponding fix lands, remove the xfail marker.

See docs/external_review_2026_09.md for the full audit.
"""

from __future__ import annotations

import numpy as np
import pytest

from scripts import compare_baselines as cb

# ------------------------------------------------------------------
# Fold construction
# ------------------------------------------------------------------


def _contiguous_layout(n_windows: int, window_size: int, stride: int) -> list[set[int]]:
    """Reconstruct the frame-id sets for a one-video contiguous layout.

    This matches the layout the harness builds from a single video with
    the given window and stride. Used to test splitter invariants without
    loading real data.
    """
    return [set(range(i * stride, i * stride + window_size)) for i in range(n_windows)]


def _leaked_validation_windows(
    folds: list[np.ndarray], frame_ids: list[set[int]]
) -> int:
    """Count validation windows that share any frame with any training window.

    A correctly-blocked split with a purge gap of at least window_size
    frames should return 0 for any fold.
    """
    leaked = 0
    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        train_frames: set[int] = set().union(*(frame_ids[i] for i in train_idx))
        for i in val_idx:
            if frame_ids[i] & train_frames:
                leaked += 1
    return leaked


@pytest.mark.xfail(
    reason="blocked_folds has no purge gap; see external_review_2026_09.md #1",
    strict=False,
)
@pytest.mark.parametrize("num_folds", [5, 10])
def test_blocked_folds_no_leaked_validation_windows(num_folds: int) -> None:
    """With overlapping windows, folds must not share frames across the
    train/val boundary. Requires a purge gap of >= window_size frames.
    """
    frame_ids = _contiguous_layout(n_windows=249, window_size=5, stride=2)
    folds = cb.blocked_folds(frame_ids, num_folds)
    leaked = _leaked_validation_windows(folds, frame_ids)
    assert leaked == 0, (
        f"num_folds={num_folds}: {leaked} leaked validation windows "
        "(folds share frames across the boundary)"
    )


def test_blocked_folds_covers_every_window_once() -> None:
    """Every window must appear in exactly one fold."""
    frame_ids = _contiguous_layout(n_windows=249, window_size=5, stride=2)
    for num_folds in (5, 10):
        folds = cb.blocked_folds(frame_ids, num_folds)
        flat = np.concatenate(folds)
        assert len(flat) == len(frame_ids)
        assert sorted(flat.tolist()) == list(range(len(frame_ids)))


# ------------------------------------------------------------------
# Tabular helper return contract
# ------------------------------------------------------------------


@pytest.mark.xfail(
    reason="train_tabular_fold returns y_train as y_val; see external_review_2026_09.md #3",
    strict=False,
)
def test_train_tabular_fold_returns_scores_aligned_with_val_size() -> None:
    """The second return value must have the same length as the first.

    Currently train_tabular_fold returns (scores, y_train), so when val
    size differs from train size the tuple is misaligned.
    """
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(12, 3)).astype(np.float64)
    y_train = np.array([0, 1] * 6, dtype=np.int64)
    X_val = rng.normal(size=(4, 3)).astype(np.float64)

    scores, returned = cb.train_tabular_fold("logreg", X_train, y_train, X_val, seed=0)

    assert len(scores) == len(returned), (
        f"scores length {len(scores)} != returned labels length {len(returned)}"
    )
    assert len(scores) == len(X_val)


# ------------------------------------------------------------------
# Threshold chooser objectives
# ------------------------------------------------------------------


def test_choose_threshold_accuracy_objective_maximizes_calibration_accuracy() -> None:
    """With objective=accuracy, the chosen threshold must maximize
    accuracy on the given (scores, labels) pair."""
    scores = np.array([0.05, 0.15, 0.25, 0.55, 0.65, 0.95], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)

    t = cb._choose_threshold(scores, labels, objective="accuracy")
    chosen_acc = ((scores >= t).astype(int) == labels).mean()
    best_acc = max(
        ((scores >= c).astype(int) == labels).mean() for c in np.unique(scores)
    )
    assert chosen_acc == best_acc, (
        f"accuracy at chosen threshold {t:.3f} is {chosen_acc:.3f}, "
        f"best possible is {best_acc:.3f}"
    )


def test_choose_threshold_youden_objective_maximizes_youden_j() -> None:
    """With objective=youden, the chosen threshold must maximize
    TPR - FPR on the given (scores, labels) pair."""
    scores = np.array([0.05, 0.15, 0.25, 0.55, 0.65, 0.95], dtype=np.float64)
    labels = np.array([0, 0, 1, 1, 0, 1], dtype=np.int64)

    t = cb._choose_threshold(scores, labels, objective="youden")

    def youden(th: float) -> float:
        preds = (scores >= th).astype(int)
        tp = int(((preds == 1) & (labels == 1)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        p = int((labels == 1).sum())
        n = int((labels == 0).sum())
        return (tp / p if p else 0.0) - (fp / n if n else 0.0)

    chosen_j = youden(t)
    best_j = max(youden(c) for c in np.unique(scores))
    assert chosen_j == best_j, (
        f"Youden J at chosen threshold {t:.3f} is {chosen_j:.3f}, "
        f"best possible is {best_j:.3f}"
    )


def test_choose_threshold_single_class_returns_half() -> None:
    """Calibration on a single-class set has no signal; must return 0.5."""
    scores = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    labels = np.array([1, 1, 1], dtype=np.int64)
    assert cb._choose_threshold(scores, labels, objective="accuracy") == 0.5
    assert cb._choose_threshold(scores, labels, objective="youden") == 0.5


def test_choose_threshold_unknown_objective_raises() -> None:
    scores = np.array([0.1, 0.9], dtype=np.float64)
    labels = np.array([0, 1], dtype=np.int64)
    with pytest.raises(ValueError):
        cb._choose_threshold(scores, labels, objective="bogus")


# ------------------------------------------------------------------
# Purge-gap invariants (external review 2026-09.md #1)
# ------------------------------------------------------------------


def test_leakage_report_raw_split_is_positive() -> None:
    """The raw blocked split leaks: adjacent windows share frames."""
    frame_ids = _contiguous_layout(n_windows=249, window_size=5, stride=2)
    folds = cb.blocked_folds(frame_ids, 10)
    rep = cb.leakage_report(folds, frame_ids, gap=0)
    assert rep["total"] == sum(rep["per_fold"])
    assert rep["num_val"] == 249
    assert rep["total"] > 0, "raw split unexpectedly leak-free"
    assert rep["gap"] == 0


def test_purge_train_indices_removes_leakage() -> None:
    """After purging with gap=window_size, no val window shares frames
    (or sits within ``gap`` frames) of any training window."""
    window_size = 5
    frame_ids = _contiguous_layout(n_windows=249, window_size=window_size, stride=2)
    folds = cb.blocked_folds(frame_ids, 10)
    rep = cb.leakage_report(folds, frame_ids, gap=window_size)
    assert rep["total"] == 0, (
        f"expected zero leakage after purge, got {rep['per_fold']}"
    )
    assert rep["gap"] == window_size
    assert all(n > 0 for n in rep["train_size_per_fold"])


def test_purge_train_indices_floor_guard() -> None:
    """If purge would empty the train set, raw train is returned."""
    frame_ids = [{0, 1, 2}, {1, 2, 3}, {100, 101, 102}]
    train = np.array([0, 1], dtype=int)
    val = np.array([2], dtype=int)
    # gap huge -> everything overlaps -> fallback to raw train
    out = cb.purge_train_indices(train, val, frame_ids, gap=10_000)
    assert sorted(out.tolist()) == [0, 1]


def test_purge_train_indices_empty_frame_windows_kept() -> None:
    """Windows with no frame_ids are kept (cannot be purged)."""
    # Window 0 shares frame 2 with the val window; window 1 has no frames.
    frame_ids = [{0, 1, 2}, set(), {2, 3, 4}]
    train = np.array([0, 1], dtype=int)
    val = np.array([2], dtype=int)
    out = cb.purge_train_indices(train, val, frame_ids, gap=0)
    assert 1 in out.tolist(), "empty-frames window must be kept"
    assert 0 not in out.tolist(), "overlapping window must be dropped"


# ------------------------------------------------------------------
# GraphFeatureExtractor feature naming and single-feature baseline
# ------------------------------------------------------------------


def test_feature_names_match_vector_width() -> None:
    """feature_names() must align with transform() output width."""
    import torch
    from torch_geometric.data import Data

    from graf.models.baselines import GraphFeatureExtractor

    g = Data(
        x=torch.zeros((3, 4), dtype=torch.float32),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15), dtype=torch.float32),
        pos=torch.zeros((3, 2), dtype=torch.float32),
    )
    vec = GraphFeatureExtractor.transform(g)
    names = GraphFeatureExtractor.feature_names()
    assert vec.shape[1] == len(names), (
        f"vector width {vec.shape[1]} != feature_names length {len(names)}"
    )
    assert "edge_attr_nonzero_frac" in names


def test_transform_excluding_drops_named_columns() -> None:
    import torch
    from torch_geometric.data import Data

    from graf.models.baselines import GraphFeatureExtractor

    g = Data(
        x=torch.zeros((3, 4)),
        edge_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        edge_attr=torch.zeros((2, 15)),
        pos=torch.zeros((3, 2)),
    )
    full = GraphFeatureExtractor.transform(g)
    reduced = GraphFeatureExtractor.transform_excluding(g, {"edge_attr_nonzero_frac"})
    assert reduced.shape[1] == full.shape[1] - 1

    with pytest.raises(ValueError):
        GraphFeatureExtractor.transform_excluding(g, {"not_a_feature"})


def test_single_feature_baseline_auc_matches_feature() -> None:
    """SingleFeatureBaseline AUC equals AUC of the chosen column."""
    from graf.models.baselines import GraphFeatureExtractor, SingleFeatureBaseline
    from scripts.evaluate_vntraffic import _auc

    rng = np.random.default_rng(0)
    n = 40
    names = GraphFeatureExtractor.feature_names()
    X = np.zeros((n, len(names)), dtype=np.float32)
    y = (rng.random(n) < 0.5).astype(int)
    X[:, names.index("edge_attr_nonzero_frac")] = y.astype(float) + rng.normal(
        scale=0.3, size=n
    )
    for j in range(X.shape[1]):
        if names[j] != "edge_attr_nonzero_frac":
            X[:, j] = rng.normal(size=n)

    clf = SingleFeatureBaseline("edge_attr_nonzero_frac").fit(X, y)
    scores = clf.predict_proba(X)[:, 1]
    auc_baseline = _auc(scores, y)
    auc_feature = _auc(X[:, names.index("edge_attr_nonzero_frac")], y)
    assert abs(auc_baseline - auc_feature) < 1e-9, (
        f"baseline AUC {auc_baseline:.6f} != feature AUC {auc_feature:.6f}"
    )
    assert auc_baseline > 0.7


# ------------------------------------------------------------------
# Harness hygiene: majority baseline (#6), threshold boundary (#7)
# ------------------------------------------------------------------


def test_majority_baseline_accuracy_uses_train_majority() -> None:
    """The majority baseline predicts the TRAIN fold's majority class.

    When train and val have different majorities, accuracy drops below
    1.0 — unlike the oracle max(p, 1-p) this replaced.
    """
    train = np.array([1, 1, 1, 1, 0])  # 80% positive
    val = np.array([0, 0, 0, 1, 1])  # 60% negative
    acc = cb._majority_baseline_accuracy(train, val)
    # Predicts positive on val → matches at positions 3, 4 → 2/5
    assert acc == 0.4


def test_majority_baseline_matches_val_majority_when_consistent() -> None:
    """When train majority applies to val, the baseline gets val accuracy."""
    train = np.array([1, 1, 1, 0])
    val = np.array([1, 1, 0, 0])
    acc = cb._majority_baseline_accuracy(train, val)
    assert acc == 0.5


def test_majority_baseline_empty_inputs_return_zero() -> None:
    assert cb._majority_baseline_accuracy(np.array([]), np.array([1, 0])) == 0.0
    assert cb._majority_baseline_accuracy(np.array([1, 0]), np.array([])) == 0.0


def test_no_strict_greater_than_0_5_in_prediction_path() -> None:
    """Threshold convention is >= 0.5 everywhere.

    _choose_threshold and calibrated accuracy use >=; the headline and
    pooled accuracy must not use > or a score of exactly 0.5 lands in
    different classes depending on the code path.
    """
    from pathlib import Path

    src = Path(__file__).resolve().parents[1] / "scripts" / "compare_baselines.py"
    text = src.read_text()
    forbidden = ["(scores > 0.5)", "(pooled_scores_arr > 0.5)"]
    for pat in forbidden:
        assert pat not in text, (
            f"{pat!r} found; use >= 0.5 for consistency with _choose_threshold"
        )


# ------------------------------------------------------------------
# Bootstrap confidence intervals
# ------------------------------------------------------------------


def test_bootstrap_ci_contains_mean_on_symmetric_data() -> None:
    """For a symmetric set, the 95% percentile CI contains the mean."""
    rng = np.random.default_rng(0)
    values = rng.normal(loc=0.75, scale=0.05, size=10)
    ci = cb._bootstrap_ci(values, n_resamples=2000, seed=42)
    assert ci is not None
    mean = float(np.mean(values))
    assert ci["lo"] <= mean <= ci["hi"], (
        f"CI [{ci['lo']:.4f}, {ci['hi']:.4f}] does not contain mean {mean:.4f}"
    )
    assert ci["lo"] < ci["hi"]


def test_bootstrap_ci_narrows_with_more_samples() -> None:
    """The CI width shrinks as the number of folds grows (identical data scale)."""
    rng = np.random.default_rng(1)
    small = rng.normal(loc=0.7, scale=0.05, size=5)
    large = rng.normal(loc=0.7, scale=0.05, size=50)
    ci_small = cb._bootstrap_ci(small, n_resamples=4000, seed=42)
    ci_large = cb._bootstrap_ci(large, n_resamples=4000, seed=42)
    assert ci_small and ci_large
    width_small = ci_small["hi"] - ci_small["lo"]
    width_large = ci_large["hi"] - ci_large["lo"]
    assert width_large < width_small, (
        f"CI width did not shrink: n=5 -> {width_small:.4f}, n=50 -> {width_large:.4f}"
    )


def test_bootstrap_ci_disabled_returns_none() -> None:
    """n_resamples=0 disables the CI and returns None."""
    assert cb._bootstrap_ci([0.7, 0.8, 0.75], n_resamples=0) is None


def test_bootstrap_ci_single_value_returns_none() -> None:
    """Fewer than two finite values cannot be resampled."""
    assert cb._bootstrap_ci([0.7], n_resamples=1000) is None
    assert cb._bootstrap_ci([], n_resamples=1000) is None
    assert cb._bootstrap_ci([float("nan"), float("nan")], n_resamples=1000) is None


def test_bootstrap_ci_seed_is_deterministic() -> None:
    """Same seed → identical CI bounds; different seed → (very likely) different."""
    values = [0.70, 0.75, 0.80, 0.72, 0.68]
    a = cb._bootstrap_ci(values, n_resamples=2000, seed=42)
    b = cb._bootstrap_ci(values, n_resamples=2000, seed=42)
    c = cb._bootstrap_ci(values, n_resamples=2000, seed=7)
    assert a == b, "same seed produced different CIs"
    assert a != c, "different seeds produced identical CIs (suspicious)"
