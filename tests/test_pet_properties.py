"""Property-based tests for ``graf.ssm.pet``.

Complements ``tests/test_property_based.py``, which already asserts PET
non-negativity, severity bounds, criticality, symmetry-in-actors,
determinism, empty-trajectory status, and PETCalculator column
validation. The invariants here are the ones the existing file does
*not* cover:

* time-shift invariance — PET is a gap, not an absolute time,
* scene-translation invariance — PET must be frame-independent,
* the ``zone_overlap`` status contract (pet_seconds is exactly 0.0),
* ``enters_first`` / ``status`` consistency,
* monotone shrinking of PET as the conflict-zone radius grows,
* ``conflict_point`` is the zone center whenever PET is finite.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

pet_mod = pytest.importorskip("graf.ssm.pet")
compute_pet_from_conflict_zone = pet_mod.compute_pet_from_conflict_zone
PETCalculator = pet_mod.PETCalculator

_coord = st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False)
_radius = st.floats(0.5, 10.0, allow_nan=False, allow_infinity=False)
_delta = st.floats(0.5, 20.0, allow_nan=False, allow_infinity=False)

_KNOWN_STATUSES = frozenset(
    {
        "uncomputed",
        "empty_trajectory",
        "one_or_both_never_enter_zone",
        "agent1_then_agent2",
        "agent2_then_agent1",
        "zone_overlap",
    }
)


@st.composite
def straight_trajectory(draw, n_min=2, n_max=10):
    """Return (points (N,2), timestamps (N,)) along a straight line.

    Timestamps are strictly increasing with unit spacing starting from a
    sampled t0 in [0, 20).
    """
    n = draw(st.integers(n_min, n_max))
    t0 = draw(st.floats(0.0, 20.0, allow_nan=False, allow_infinity=False))
    ts = np.array([t0 + i for i in range(n)], dtype=float)
    x0 = draw(_coord)
    y0 = draw(_coord)
    dx = draw(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    dy = draw(st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False))
    pts = np.array([[x0 + i * dx, y0 + i * dy] for i in range(n)], dtype=float)
    return pts, ts


# ── Invariances ──────────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(
    t1=straight_trajectory(),
    t2=straight_trajectory(),
    cx=_coord,
    cy=_coord,
    r=_radius,
    dt=_delta,
)
@settings(max_examples=50, deadline=None)
def test_pet_invariant_under_time_shift(t1, t2, cx, cy, r, dt):
    """Shifting both timestamp arrays by a constant leaves PET unchanged."""
    p1, ts1 = t1
    p2, ts2 = t2
    center = np.array([cx, cy])
    base = compute_pet_from_conflict_zone(p1, p2, ts1, ts2, center, r)
    shifted = compute_pet_from_conflict_zone(
        p1, p2, ts1 + dt, ts2 + dt, center, r
    )
    if math.isinf(base.pet_seconds):
        assert math.isinf(shifted.pet_seconds)
    else:
        assert shifted.pet_seconds == pytest.approx(base.pet_seconds, abs=1e-9)
        assert shifted.status == base.status
        assert shifted.enters_first == base.enters_first


@pytest.mark.hypothesis
@given(
    t1=straight_trajectory(),
    t2=straight_trajectory(),
    cx=_coord,
    cy=_coord,
    r=_radius,
    dx=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
    dy=st.floats(-20.0, 20.0, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=50, deadline=None)
def test_pet_invariant_under_scene_translation(t1, t2, cx, cy, r, dx, dy):
    """Translating tracks and zone together leaves PET unchanged."""
    p1, ts1 = t1
    p2, ts2 = t2
    base = compute_pet_from_conflict_zone(
        p1, p2, ts1, ts2, np.array([cx, cy]), r
    )
    moved = compute_pet_from_conflict_zone(
        p1 + np.array([dx, dy]),
        p2 + np.array([dx, dy]),
        ts1,
        ts2,
        np.array([cx + dx, cy + dy]),
        r,
    )
    if math.isinf(base.pet_seconds):
        assert math.isinf(moved.pet_seconds)
    else:
        assert moved.pet_seconds == pytest.approx(base.pet_seconds, abs=1e-9)


# ── Status / enters_first contracts ──────────────────────────────────
@pytest.mark.hypothesis
@given(t1=straight_trajectory(), t2=straight_trajectory(), cx=_coord, cy=_coord, r=_radius)
@settings(max_examples=50, deadline=None)
def test_pet_status_is_in_known_enum(t1, t2, cx, cy, r):
    p1, ts1 = t1
    p2, ts2 = t2
    result = compute_pet_from_conflict_zone(
        p1, p2, ts1, ts2, np.array([cx, cy]), r
    )
    assert result.status in _KNOWN_STATUSES


@pytest.mark.hypothesis
@given(t1=straight_trajectory(), t2=straight_trajectory(), cx=_coord, cy=_coord, r=_radius)
@settings(max_examples=50, deadline=None)
def test_pet_zone_overlap_yields_exactly_zero(t1, t2, cx, cy, r):
    """The zone_overlap status implies pet_seconds == 0.0 exactly."""
    p1, ts1 = t1
    p2, ts2 = t2
    result = compute_pet_from_conflict_zone(
        p1, p2, ts1, ts2, np.array([cx, cy]), r
    )
    if result.status == "zone_overlap":
        assert result.pet_seconds == 0.0
        assert result.enters_first is None


@pytest.mark.hypothesis
@given(t1=straight_trajectory(), t2=straight_trajectory(), cx=_coord, cy=_coord, r=_radius)
@settings(max_examples=50, deadline=None)
def test_pet_enters_first_consistent_with_status(t1, t2, cx, cy, r):
    """status -> enters_first contract."""
    p1, ts1 = t1
    p2, ts2 = t2
    result = compute_pet_from_conflict_zone(
        p1, p2, ts1, ts2, np.array([cx, cy]), r
    )
    mapping = {
        "agent1_then_agent2": "agent1",
        "agent2_then_agent1": "agent2",
        "zone_overlap": None,
        "one_or_both_never_enter_zone": None,
        "empty_trajectory": None,
        "uncomputed": None,
    }
    assert result.enters_first == mapping[result.status]


# ── Monotonicity in the zone radius ──────────────────────────────────
@pytest.mark.hypothesis
@given(t1=straight_trajectory(), t2=straight_trajectory(), cx=_coord, cy=_coord)
@settings(max_examples=50, deadline=None)
def test_pet_monotone_decreasing_in_zone_radius(t1, t2, cx, cy):
    """Growing the conflict zone can only shrink PET.

    Enlarging the zone makes the first actor exit later and the second
    actor enter earlier, so the gap between the two events can only
    shrink — until the two overlap in the zone (PET = 0).
    """
    p1, ts1 = t1
    p2, ts2 = t2
    center = np.array([cx, cy])
    small = compute_pet_from_conflict_zone(p1, p2, ts1, ts2, center, 1.0)
    big = compute_pet_from_conflict_zone(p1, p2, ts1, ts2, center, 25.0)
    if math.isfinite(small.pet_seconds):
        # A larger concentric zone contains the smaller one, so if the
        # small zone was entered, the big zone is entered too.
        assert math.isfinite(big.pet_seconds)
        assert big.pet_seconds <= small.pet_seconds + 1e-9


# ── conflict_point contract ──────────────────────────────────────────
@pytest.mark.hypothesis
@given(t1=straight_trajectory(), t2=straight_trajectory(), cx=_coord, cy=_coord, r=_radius)
@settings(max_examples=50, deadline=None)
def test_pet_conflict_point_is_zone_center_when_finite(t1, t2, cx, cy, r):
    p1, ts1 = t1
    p2, ts2 = t2
    result = compute_pet_from_conflict_zone(
        p1, p2, ts1, ts2, np.array([cx, cy]), r
    )
    if math.isfinite(result.pet_seconds):
        assert result.conflict_point is not None
        assert result.conflict_point[0] == pytest.approx(cx)
        assert result.conflict_point[1] == pytest.approx(cy)


# ── PETCalculator: frame ordering ────────────────────────────────────
def _pair_df(frame_idxs, x_offsets, track_id):
    n = len(frame_idxs)
    return pd.DataFrame(
        {
            "track_id": np.full(n, track_id, dtype=int),
            "frame_idx": np.asarray(frame_idxs, dtype=int),
            "t_sec": np.asarray(frame_idxs, dtype=float) * 0.1,
            "x_m": np.asarray(x_offsets, dtype=float),
            "y_m": np.zeros(n, dtype=float),
        }
    )


@pytest.mark.hypothesis
@given(n=st.integers(2, 10), gap=st.floats(0.0, 0.5))
@settings(max_examples=30, deadline=None)
def test_pet_calculator_event_frames_are_ordered(n, gap):
    frames = np.arange(n, dtype=int)
    df_a = _pair_df(frames, np.zeros(n), track_id=1)
    df_b = _pair_df(frames, np.full(n, gap), track_id=2)
    calc = PETCalculator(proximity_threshold_m=5.0, critical_threshold_s=5.0)
    ev = calc.compute_pair_pet(df_a, df_b, video_id="v")
    if ev is not None:
        assert ev.end_frame >= ev.start_frame


@pytest.mark.hypothesis
@given(n=st.integers(2, 8), gap=st.floats(5.0, 50.0))
@settings(max_examples=20, deadline=None)
def test_pet_calculator_far_pair_returns_none(n, gap):
    frames = np.arange(n, dtype=int)
    df_a = _pair_df(frames, np.zeros(n), track_id=1)
    df_b = _pair_df(frames, np.full(n, gap), track_id=2)
    calc = PETCalculator(proximity_threshold_m=1.0, critical_threshold_s=5.0)
    ev = calc.compute_pair_pet(df_a, df_b, video_id="v")
    assert ev is None
