"""Tests for scripts/label_strategies.py."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import label_strategies as ls

# ------------------------------------------------------------------
# Test fixture: a light stand-in for SpatioTemporalWindowDataset
# ------------------------------------------------------------------


class _FakeWindow:
    def __init__(self, frame_ids: list[int]) -> None:
        self.frame_ids = np.asarray(frame_ids, dtype=int)


class _FakeDS:
    """SpatioTemporalWindowDataset-shaped: __len__, __getitem__ with
    a ``frame_ids`` attribute."""

    def __init__(self, windows: list[list[int]]) -> None:
        self._windows = [_FakeWindow(w) for w in windows]

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, i: int) -> _FakeWindow:
        return self._windows[i]


def _df(rows: list[tuple]) -> pd.DataFrame:
    """Build a tracks DataFrame with the columns label_strategies uses."""
    return pd.DataFrame(
        rows,
        columns=[
            "frame_idx",
            "track_id",
            "x_m",
            "y_m",
            "vx",
            "vy",
        ],
    )


# ------------------------------------------------------------------
# _frame_ttc_stats
# ------------------------------------------------------------------


def test_frame_ttc_stats_no_pairs():
    # Single actor => no pairs.
    df = _df([(0, 1, 0.0, 0.0, 1.0, 0.0)])
    stats = ls._frame_ttc_stats(
        df,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
    )
    assert stats.loc[0, "n_pairs"] == 0
    assert stats.loc[0, "n_critical"] == 0
    assert np.isinf(stats.loc[0, "min_ttc"])


def test_frame_ttc_stats_single_critical_pair():
    # Two actors 2 m apart closing at 1 m/s each => TTC = 1.0 s.
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    stats = ls._frame_ttc_stats(
        df,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
    )
    assert stats.loc[0, "n_pairs"] == 1
    assert stats.loc[0, "n_critical"] == 1
    assert stats.loc[0, "min_ttc"] == pytest.approx(1.0)


def test_frame_ttc_stats_pair_outside_distance_not_counted():
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 10.0, 0.0, -1.0, 0.0),  # 10 m apart > 3.0 m threshold
        ]
    )
    stats = ls._frame_ttc_stats(
        df,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
    )
    assert stats.loc[0, "n_pairs"] == 0


# ------------------------------------------------------------------
# Strategy: any
# ------------------------------------------------------------------


def test_label_any_basic():
    df = _df(
        [
            # frame 0: critical (TTC = 1.0)
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
            # frame 1: not critical (diverge)
            (1, 1, 0.0, 0.0, -1.0, 0.0),
            (1, 2, 2.0, 0.0, 1.0, 0.0),
            # frame 2: critical
            (2, 1, 0.0, 0.0, 1.0, 0.0),
            (2, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0, 1, 2], [1, 2, 0], [0, 1, 1]])
    labels = ls.label_any(df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0)
    assert labels == [1, 1, 1]  # all three windows have frame 0 or 2


def test_label_any_no_critical_frames():
    df = _df(
        [
            (0, 1, 0.0, 0.0, -1.0, 0.0),
            (0, 2, 2.0, 0.0, 1.0, 0.0),  # diverge
        ]
    )
    ds = _FakeDS([[0]])
    labels = ls.label_any(df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0)
    assert labels == [0]


# ------------------------------------------------------------------
# Strategy: pair_fraction
# ------------------------------------------------------------------


def test_label_pair_fraction_empty_frame_is_negative():
    # Single actor => 0 pairs => fraction 0 => always negative
    df = _df([(0, 1, 0.0, 0.0, 1.0, 0.0)])
    ds = _FakeDS([[0]])
    labels = ls.label_pair_fraction(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        fraction_threshold=0.1,
    )
    assert labels == [0]


def test_label_pair_fraction_high_threshold_requires_many_critical():
    # 4 actors in a tight cluster: pairs=6. One critical pair.
    # Fraction = 1/6 ≈ 0.167. fraction_threshold=0.5 -> negative.
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),  # critical with #1
            (0, 3, 0.0, 1.0, 0.0, 0.0),  # nearby, no approach
            (0, 4, 0.0, -1.0, 0.0, 0.0),  # nearby, no approach
        ]
    )
    ds = _FakeDS([[0]])
    labels = ls.label_pair_fraction(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        fraction_threshold=0.5,
    )
    assert labels == [0]

    labels_low = ls.label_pair_fraction(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        fraction_threshold=0.1,
    )
    assert labels_low == [1]


# ------------------------------------------------------------------
# Strategy: sustained
# ------------------------------------------------------------------


def test_label_sustained_requires_consecutive_run():
    # Frames 0, 2 critical; frame 1 not. Longest run of consecutive
    # critical frames is 1.
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
            (1, 1, 0.0, 0.0, -1.0, 0.0),
            (1, 2, 2.0, 0.0, 1.0, 0.0),  # diverge
            (2, 1, 0.0, 0.0, 1.0, 0.0),
            (2, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0, 1, 2]])
    # run_length=2 -> no two consecutive critical -> negative
    assert ls.label_sustained(
        df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0, run_length=2
    ) == [0]
    # run_length=1 -> matches "any"
    assert ls.label_sustained(
        df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0, run_length=1
    ) == [1]


def test_label_sustained_run_of_three():
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
            (1, 1, 0.0, 0.0, 1.0, 0.0),
            (1, 2, 2.0, 0.0, -1.0, 0.0),
            (2, 1, 0.0, 0.0, 1.0, 0.0),
            (2, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0, 1, 2]])
    assert ls.label_sustained(
        df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0, run_length=3
    ) == [1]


def test_label_sustained_rejects_zero_run_length():
    df = _df([(0, 1, 0.0, 0.0, 1.0, 0.0)])
    ds = _FakeDS([[0]])
    with pytest.raises(ValueError, match="run_length"):
        ls.label_sustained(df, ds, run_length=0)


# ------------------------------------------------------------------
# Strategy: min_ttc
# ------------------------------------------------------------------


def test_label_min_ttc_stricter_threshold_is_negative():
    # TTC = 1.0 s; threshold 0.5 s -> negative; threshold 1.5 s -> positive
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0]])
    assert ls.label_min_ttc(
        df, ds, ttc_threshold_seconds=0.5, distance_threshold=3.0
    ) == [0]
    assert ls.label_min_ttc(
        df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0
    ) == [1]


def test_label_min_ttc_matches_any_at_frame_level():
    """min_ttc == any when the same threshold is used at the frame level."""
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0]])
    a = ls.label_any(df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0)
    b = ls.label_min_ttc(df, ds, ttc_threshold_seconds=1.5, distance_threshold=3.0)
    assert a == b == [1]


# ------------------------------------------------------------------
# apply_strategy dispatch
# ------------------------------------------------------------------


def test_apply_strategy_dispatches():
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0]])
    labels = ls.apply_strategy(
        "any",
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
    )
    assert labels == [1]


def test_apply_strategy_unknown_raises():
    df = _df([(0, 1, 0.0, 0.0, 1.0, 0.0)])
    ds = _FakeDS([[0]])
    with pytest.raises(ValueError, match="unknown label strategy"):
        ls.apply_strategy("not-a-strategy", df, ds)


def test_apply_strategy_forwards_kwargs():
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    ds = _FakeDS([[0]])
    labels = ls.apply_strategy(
        "sustained",
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        run_length=2,
    )
    # Single-frame window, run_length=2 cannot be met
    assert labels == [0]
