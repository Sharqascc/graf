"""Metamorphic tests: input transformations must transform outputs predictably.

We do not always know the "correct" output for a given input (TTC, DRAC,
homography). But we do know how the output must change when the input is
transformed in a way the algorithm is invariant to. If the transformation
rule is violated, there is a bug.

Cases below:

  * TTC is invariant to translation and rotation of both actors.
  * DRAC is invariant to translation and rotation of both actors.
  * Kinematics velocities are invariant to position translation.
  * A homography that is a pure image translation produces a pure world
    translation (same offset for every point).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graf.calibration.homography import project_points
from graf.ssm.drac import compute_drac_constant_velocity
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.kinematics import compute_kinematics

_point = st.tuples(
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
    st.floats(-100.0, 100.0, allow_nan=False, allow_infinity=False),
)
_vel = st.tuples(
    st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
    st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
)
_shift = st.tuples(
    st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
    st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False),
)


# ------------------------------------------------------------------
# TTC
# ------------------------------------------------------------------


@pytest.mark.hypothesis
@given(p1=_point, v1=_vel, p2=_point, v2=_vel, shift=_shift)
@settings(max_examples=30, deadline=None)
def test_ttc_translation_invariant(p1, v1, p2, v2, shift):
    a = compute_ttc_constant_velocity(p1, v1, p2, v2)
    p1s = (p1[0] + shift[0], p1[1] + shift[1])
    p2s = (p2[0] + shift[0], p2[1] + shift[1])
    b = compute_ttc_constant_velocity(p1s, v1, p2s, v2)
    if np.isfinite(a.ttc_seconds) or np.isfinite(b.ttc_seconds):
        assert np.isclose(a.ttc_seconds, b.ttc_seconds, rtol=1e-5, atol=1e-6)
    assert a.status == b.status


@pytest.mark.hypothesis
@given(
    p1=_point,
    v1=_vel,
    p2=_point,
    v2=_vel,
    theta=st.floats(0.0, 2.0 * np.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_ttc_rotation_invariant(p1, v1, p2, v2, theta):
    def rot(xy):
        c, s = np.cos(theta), np.sin(theta)
        return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])

    a = compute_ttc_constant_velocity(p1, v1, p2, v2)
    b = compute_ttc_constant_velocity(rot(p1), rot(v1), rot(p2), rot(v2))
    if np.isfinite(a.ttc_seconds) or np.isfinite(b.ttc_seconds):
        assert np.isclose(a.ttc_seconds, b.ttc_seconds, rtol=1e-4, atol=1e-5)
    assert a.status == b.status


# ------------------------------------------------------------------
# DRAC
# ------------------------------------------------------------------


@pytest.mark.hypothesis
@given(p1=_point, v1=_vel, p2=_point, v2=_vel, shift=_shift)
@settings(max_examples=30, deadline=None)
def test_drac_translation_invariant(p1, v1, p2, v2, shift):
    a = compute_drac_constant_velocity(p1, v1, p2, v2)
    p1s = (p1[0] + shift[0], p1[1] + shift[1])
    p2s = (p2[0] + shift[0], p2[1] + shift[1])
    b = compute_drac_constant_velocity(p1s, v1, p2s, v2)
    if np.isfinite(a.drac_mps2) and np.isfinite(b.drac_mps2):
        assert np.isclose(a.drac_mps2, b.drac_mps2, rtol=1e-5, atol=1e-6)
    assert a.status == b.status


@pytest.mark.hypothesis
@given(
    p1=_point,
    v1=_vel,
    p2=_point,
    v2=_vel,
    theta=st.floats(0.0, 2.0 * np.pi, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_drac_rotation_invariant(p1, v1, p2, v2, theta):
    def rot(xy):
        c, s = np.cos(theta), np.sin(theta)
        return (c * xy[0] - s * xy[1], s * xy[0] + c * xy[1])

    a = compute_drac_constant_velocity(p1, v1, p2, v2)
    b = compute_drac_constant_velocity(rot(p1), rot(v1), rot(p2), rot(v2))
    if np.isfinite(a.drac_mps2) and np.isfinite(b.drac_mps2):
        assert np.isclose(a.drac_mps2, b.drac_mps2, rtol=1e-4, atol=1e-5)
    assert a.status == b.status


# ------------------------------------------------------------------
# Kinematics
# ------------------------------------------------------------------


def test_kinematics_translation_invariant():
    """Shifting every position by (dx, dy) leaves velocities unchanged."""
    n = 20
    base = pd.DataFrame(
        {
            "track_id": [1] * n,
            "frame_idx": list(range(n)),
            "x_m": np.linspace(0.0, 10.0, n),
            "y_m": np.linspace(0.0, 5.0, n),
        }
    )
    shifted = base.copy()
    shifted["x_m"] += 100.0
    shifted["y_m"] -= 50.0

    a = compute_kinematics(base, fps=30.0)
    b = compute_kinematics(shifted, fps=30.0)
    pd.testing.assert_series_equal(a["vx_mps"], b["vx_mps"])
    pd.testing.assert_series_equal(a["vy_mps"], b["vy_mps"])
    pd.testing.assert_series_equal(a["speed_mps"], b["speed_mps"])


# ------------------------------------------------------------------
# Homography
# ------------------------------------------------------------------


def test_homography_pure_translation_preserves_offset():
    """A homography that is a pure image translation produces a pure world
    translation — every projected point shifts by the same vector."""
    H = np.eye(3)
    H[0, 2] = 10.0
    H[1, 2] = -5.0

    points1 = [(0.0, 0.0), (5.0, 3.0), (100.0, 50.0), (-20.0, 80.0)]
    points2 = [(x + 10.0, y - 5.0) for (x, y) in points1]

    w1 = project_points(H, points1)
    w2 = project_points(H, points2)

    delta = w2 - w1
    expected = np.tile([10.0, -5.0], (len(points1), 1))
    np.testing.assert_allclose(delta, expected, atol=1e-9)
