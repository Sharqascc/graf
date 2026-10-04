"""Tests for the persistent_pair label strategy."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts import label_strategies as ls


class _FakeWindow:
    def __init__(self, frame_ids):
        self.frame_ids = np.asarray(frame_ids, dtype=int)


class _FakeDS:
    def __init__(self, windows):
        self._windows = [_FakeWindow(w) for w in windows]

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, i):
        return self._windows[i]


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=["frame_idx", "track_id", "x_m", "y_m", "vx", "vy"],
    )


def test_same_pair_critical_for_three_frames_labels_positive():
    """Two actors closing head-on for 3 consecutive frames. run_length=3."""
    rows = []
    for f in range(3):
        rows.append((f, 1, 0.0 + f * 0.1, 0.0, 1.0, 0.0))
        rows.append((f, 2, 1.5 - f * 0.1, 0.0, -1.0, 0.0))
    df = _df(rows)
    ds = _FakeDS([[0, 1, 2]])
    labels = ls.label_persistent_pair(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=3,
    )
    assert labels == [1]


def test_different_pairs_on_consecutive_frames_labels_negative():
    """Three frames, each critical, but with a *different* pair every
    frame. persistent_pair(run_length=2) must be 0 even though
    sustained(run_length=2) is 1 on the same data."""
    rows = [
        (0, 1, 0.0, 0.0, 1.0, 0.0),
        (0, 2, 1.5, 0.0, -1.0, 0.0),
        (1, 3, 0.0, 10.0, 1.0, 0.0),
        (1, 4, 1.5, 10.0, -1.0, 0.0),
    ]
    df = _df(rows)
    ds = _FakeDS([[0, 1]])
    labels = ls.label_persistent_pair(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=2,
    )
    assert labels == [0]

    sustained = ls.label_sustained(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=2,
    )
    assert sustained == [1]


def test_no_critical_pair_labels_negative():
    rows = [
        (0, 1, 0.0, 0.0, 0.0, 0.0),
        (0, 2, 50.0, 0.0, 0.0, 0.0),
    ]
    df = _df(rows)
    ds = _FakeDS([[0]])
    labels = ls.label_persistent_pair(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=1,
    )
    assert labels == [0]


def test_run_length_invalid_raises():
    df = _df([(0, 1, 0.0, 0.0, 1.0, 0.0)])
    ds = _FakeDS([[0]])
    with pytest.raises(ValueError):
        ls.label_persistent_pair(df, ds, run_length=0)


def test_same_pair_resets_when_interrupted():
    """Pair critical at frames 0, 2, 3 -> longest same-pair run is 2."""
    rows = [
        (0, 1, 0.0, 0.0, 1.0, 0.0),
        (0, 2, 1.5, 0.0, -1.0, 0.0),
        # frame 1: 5 m apart, above the 3 m distance threshold, so the
        # pair is not critical and any run is interrupted.
        (1, 1, 0.0, 0.0, 1.0, 0.0),
        (1, 2, 5.0, 0.0, -1.0, 0.0),
        # frame 2: back within threshold
        (2, 1, 0.0, 0.0, 1.0, 0.0),
        (2, 2, 1.5, 0.0, -1.0, 0.0),
        # frame 3: still within threshold
        (3, 1, 0.0, 0.0, 1.0, 0.0),
        (3, 2, 1.5, 0.0, -1.0, 0.0),
    ]
    df = _df(rows)
    ds = _FakeDS([[0, 1, 2, 3]])
    labels = ls.label_persistent_pair(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=2,
    )
    assert labels == [1]  # frames 2 and 3 give a run of 2

    # A run_length=3 request on the same data must fail: the
    # interruption at frame 1 breaks the sequence.
    labels3 = ls.label_persistent_pair(
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        closing_rate_threshold=0.5,
        run_length=3,
    )
    assert labels3 == [0]


def test_strategy_registered():
    assert "persistent_pair" in ls.STRATEGIES
    assert "persistent_pair" in ls._DISPATCH


def test_apply_strategy_dispatches_persistent_pair():
    rows = []
    for f in range(3):
        rows.append((f, 1, 0.0, 0.0, 1.0, 0.0))
        rows.append((f, 2, 1.5, 0.0, -1.0, 0.0))
    df = _df(rows)
    ds = _FakeDS([[0, 1, 2]])
    labels = ls.apply_strategy(
        "persistent_pair",
        df,
        ds,
        ttc_threshold_seconds=1.5,
        distance_threshold=3.0,
        run_length=3,
    )
    assert labels == [1]
