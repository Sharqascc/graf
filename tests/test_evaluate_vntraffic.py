"""Tests for scripts/evaluate_vntraffic.py."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from scripts import evaluate_vntraffic as script_mod

# --- F1 -----------------------------------------------------------


def test_f1_perfect():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([1, 1, 0, 0])
    assert script_mod._f1(preds, labels) == 1.0


def test_f1_all_wrong():
    preds = np.array([1, 1, 0, 0])
    labels = np.array([0, 0, 1, 1])
    assert script_mod._f1(preds, labels) == 0.0


def test_f1_no_positives_predicted():
    preds = np.array([0, 0, 0, 0])
    labels = np.array([1, 1, 0, 0])
    assert script_mod._f1(preds, labels) == 0.0


# --- AUC ----------------------------------------------------------


def test_auc_perfect_separation():
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert script_mod._auc(scores, labels) == 1.0


def test_auc_inverted():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    labels = np.array([1, 1, 0, 0])
    assert script_mod._auc(scores, labels) == 0.0


def test_auc_one_class_returns_nan():
    scores = np.array([0.5, 0.6])
    labels = np.array([1, 1])
    assert np.isnan(script_mod._auc(scores, labels))


def test_auc_ties_give_half():
    scores = np.array([0.5, 0.5])
    labels = np.array([1, 0])
    assert script_mod._auc(scores, labels) == 0.5


def test_auc_partial_ties():
    # Positives {0.5, 0.6} vs negatives {0.5, 0.4}. Enumerate pairs:
    #   0.5 vs 0.5 -> tie  -> 0.5
    #   0.5 vs 0.4 -> win  -> 1.0
    #   0.6 vs 0.5 -> win  -> 1.0
    #   0.6 vs 0.4 -> win  -> 1.0
    # sum = 3.5 / 4 pairs = 0.875.
    scores = np.array([0.5, 0.6, 0.5, 0.4])
    labels = np.array([1, 1, 0, 0])
    assert script_mod._auc(scores, labels) == pytest.approx(0.875)


# --- split strategies ---------------------------------------------


def test_blocked_folds_are_contiguous():
    frame_ids = [[i, i + 1] for i in range(10)]
    folds = script_mod.blocked_folds(frame_ids, num_folds=5)
    assert len(folds) == 5
    for f in folds:
        s = sorted(f.tolist())
        diffs = [b - a for a, b in itertools.pairwise(s)]
        assert all(d == 1 for d in diffs), s


def test_blocked_folds_leak_only_near_boundaries():
    """Interior val windows (>=2 positions from either end) never leak."""
    windows = [list(range(i * 2, i * 2 + 5)) for i in range(20)]
    folds = script_mod.blocked_folds(windows, num_folds=4)
    for k in range(4):
        val_idx = sorted(int(x) for x in folds[k])
        train_idx = np.concatenate([folds[j] for j in range(4) if j != k])
        train_frames = set()
        for i in train_idx:
            train_frames.update(windows[int(i)])
        for pos, v in enumerate(val_idx):
            if set(windows[v]) & train_frames:
                near_start = pos < 2
                near_end = pos >= len(val_idx) - 2
                assert near_start or near_end, (
                    f"fold {k} interior val window {v} (pos {pos}) leaked"
                )


def test_random_folds_leak_more_than_blocked():
    windows = [list(range(i * 2, i * 2 + 5)) for i in range(20)]
    blocked = script_mod.blocked_folds(windows, num_folds=4)
    random = script_mod.random_folds(20, num_folds=4, seed=42)

    def total_leaked(folds):
        total = 0
        for k in range(len(folds)):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
            n_leaked, _ = script_mod._leakage_count(train_idx, val_idx, windows)
            total += n_leaked
        return total

    assert total_leaked(random) > total_leaked(blocked)


# --- leakage counter ----------------------------------------------


def test_leakage_none_when_disjoint():
    windows = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    n_leaked, n_val = script_mod._leakage_count([0], [1, 2], windows)
    assert n_leaked == 0
    assert n_val == 2


def test_leakage_counts_shared_frames():
    windows = [[0, 1, 2], [2, 3, 4], [5, 6, 7]]
    n_leaked, n_val = script_mod._leakage_count([0], [1, 2], windows)
    assert n_leaked == 1
    assert n_val == 2


# --- binomial vs majority -----------------------------------------


def test_binomial_at_majority_not_significant():
    p = script_mod._binomial_vs_majority(correct=32, total=50, majority_rate=0.65)
    assert p > 0.4


def test_binomial_beats_majority_clearly():
    p = script_mod._binomial_vs_majority(correct=48, total=50, majority_rate=0.65)
    assert p < 0.001


def test_binomial_empty_returns_one():
    assert script_mod._binomial_vs_majority(0, 0, 0.5) == 1.0


# --- TTC positive frames -------------------------------------------


def _make_ttc_df(rows):
    import pandas as pd

    return pd.DataFrame(
        rows,
        columns=["frame_idx", "track_id", "x_m", "y_m", "vx", "vy"],
    )


def test_build_ttc_positive_frames_empty_df():
    import pandas as pd

    empty = pd.DataFrame(columns=["frame_idx", "track_id", "x_m", "y_m", "vx", "vy"])
    assert script_mod.build_ttc_positive_frames(empty) == set()


def test_build_ttc_positive_frames_head_on_collision():
    # Two actors, 10 m apart, closing at 3 + 3 = 6 m/s head-on.
    # closing_rate = 30, rel_speed_sq = 36, ttc = 30/36 ≈ 0.833 s
    # distance = 10 m < 3 m threshold? NO, 10 > 3, so no event.
    df = _make_ttc_df(
        [
            (0, "a", 0.0, 0.0, 3.0, 0.0),
            (0, "b", 10.0, 0.0, -3.0, 0.0),
        ]
    )
    assert script_mod.build_ttc_positive_frames(df) == set()


def test_build_ttc_positive_frames_close_pair():
    # Actors 2 m apart, closing head-on. Well within distance threshold.
    df = _make_ttc_df(
        [
            (0, "a", 0.0, 0.0, 1.0, 0.0),
            (0, "b", 2.0, 0.0, -1.0, 0.0),
        ]
    )
    # rel_pos = (2, 0), rel_vel = (-2, 0)
    # dist = 2, closing_rate = -dot((2,0),(-2,0)) = 4
    # rel_speed_sq = 4, ttc = 4/4 = 1.0 s <= 1.5 -> positive
    frames = script_mod.build_ttc_positive_frames(df)
    assert frames == {0}


def test_build_ttc_positive_frames_diverge_no_event():
    # Actors close but diverging.
    df = _make_ttc_df(
        [
            (0, "a", 0.0, 0.0, -1.0, 0.0),
            (0, "b", 2.0, 0.0, 1.0, 0.0),
        ]
    )
    assert script_mod.build_ttc_positive_frames(df) == set()


def test_build_ttc_threshold_boundary():
    # ttc exactly at threshold -> positive (<=)
    df = _make_ttc_df(
        [
            (0, "a", 0.0, 0.0, 1.0, 0.0),
            (0, "b", 2.0, 0.0, -1.0, 0.0),
        ]
    )
    # ttc = 1.0; threshold 1.0 -> positive
    assert script_mod.build_ttc_positive_frames(df, ttc_threshold_seconds=1.0) == {0}
    # threshold 0.99 -> not positive
    assert script_mod.build_ttc_positive_frames(df, ttc_threshold_seconds=0.99) == set()


def test_build_ttc_multiple_frames_picks_positive_ones():
    df = _make_ttc_df(
        [
            # frame 0: far apart -> no event
            (0, "a", 0.0, 0.0, 1.0, 0.0),
            (0, "b", 10.0, 0.0, -1.0, 0.0),
            # frame 1: close, closing -> event
            (1, "a", 0.0, 0.0, 1.0, 0.0),
            (1, "b", 2.0, 0.0, -1.0, 0.0),
        ]
    )
    assert script_mod.build_ttc_positive_frames(df) == {1}
