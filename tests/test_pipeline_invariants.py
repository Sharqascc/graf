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
