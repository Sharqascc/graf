"""Parity test between the label-pipeline TTC and ssm/ttc.py.

The label pipeline (`scripts/label_strategies._frame_ttc_stats`)
computes TTC inline as `closing_rate / rel_speed_sq` — time to
closest approach. `src/graf/ssm/ttc.compute_ttc_constant_velocity`
solves the collision-radius quadratic. The two are not the same
formula, and this file establishes exactly when they agree.

They agree for collinear head-on approaches, because then the
closest-approach time equals the collision time at radius zero.
They diverge for non-collinear approaches, because the label
pipeline still returns a finite closest-approach time while the
quadratic returns inf when the paths do not intersect.

This documents a real difference between the tested TTC and the
label's TTC rather than papering over it. A follow-up study could
switch the label to call `compute_ttc_constant_velocity` directly
with a collision radius; that would change the label values and
therefore the paper's frozen results, so it is not done here.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from graf.ssm.ttc import compute_ttc_constant_velocity
from scripts import label_strategies as ls


def _df(rows):
    return pd.DataFrame(
        rows,
        columns=["frame_idx", "track_id", "x_m", "y_m", "vx", "vy"],
    )


def _label_min_ttc(
    df,
    *,
    ttc_threshold_seconds=10.0,
    distance_threshold=100.0,
    closing_rate_threshold=0.0,
):
    """The label pipeline's minimum TTC across critical pairs, or inf."""
    stats = ls._frame_ttc_stats(
        df,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
    )
    return float(stats["min_ttc"].min())


def _ssm_ttc(pos_a, vel_a, pos_b, vel_b, *, min_distance=0.0):
    r = compute_ttc_constant_velocity(
        np.asarray(pos_a, dtype=float),
        np.asarray(vel_a, dtype=float),
        np.asarray(pos_b, dtype=float),
        np.asarray(vel_b, dtype=float),
        min_distance=min_distance,
    )
    return r.ttc_seconds


def test_head_on_collinear_pair_agree():
    """Collinear head-on at 2 m separation, 1 m/s closing each."""
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, -1.0, 0.0),
        ]
    )
    label_ttc = _label_min_ttc(df)
    ssm_ttc = _ssm_ttc(
        pos_a=(0.0, 0.0),
        vel_a=(1.0, 0.0),
        pos_b=(2.0, 0.0),
        vel_b=(-1.0, 0.0),
        min_distance=0.0,
    )
    assert label_ttc == pytest.approx(1.0, abs=1e-12)
    assert ssm_ttc == pytest.approx(1.0, abs=1e-12)
    assert label_ttc == pytest.approx(ssm_ttc, abs=1e-12)


def test_head_on_offset_pair_diverges():
    """Non-collinear approach: label returns closest-approach time,"
    ssm returns inf because paths do not cross."""
    # a at (0, 0) moving +x; b at (2, 1) moving -x; closest approach
    # at x=1, y=0 vs y=1 -> 1 m apart, never within 0 m.
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 1.0, -1.0, 0.0),
        ]
    )
    label_ttc = _label_min_ttc(df)
    ssm_ttc = _ssm_ttc(
        pos_a=(0.0, 0.0),
        vel_a=(1.0, 0.0),
        pos_b=(2.0, 1.0),
        vel_b=(-1.0, 0.0),
        min_distance=0.0,
    )
    # Label returns a finite closest-approach time.
    assert math.isfinite(label_ttc)
    assert label_ttc == pytest.approx(1.0, abs=1e-12)
    # SSM returns inf: no point collision on a 1-m offset course.
    assert math.isinf(ssm_ttc)


def test_ssm_with_radius_collision_agrees_with_label_for_collinear():
    """With a nonzero radius, SSM returns the time to enter the
    collision disc. For a collinear head-on pair it equals the
    closest-approach time minus the radius divided by closing speed."""
    # r = 2 m apart, 2 m/s closing total, radius 0.5 m -> entry at t=0.75 s
    label_ttc = _label_min_ttc(
        _df(
            [
                (0, 1, 0.0, 0.0, 1.0, 0.0),
                (0, 2, 2.0, 0.0, -1.0, 0.0),
            ]
        )
    )
    ssm_ttc_radius = _ssm_ttc(
        pos_a=(0.0, 0.0),
        vel_a=(1.0, 0.0),
        pos_b=(2.0, 0.0),
        vel_b=(-1.0, 0.0),
        min_distance=0.5,
    )
    assert label_ttc == pytest.approx(1.0, abs=1e-12)
    assert ssm_ttc_radius == pytest.approx(0.75, abs=1e-12)


@pytest.mark.parametrize(
    "theta_deg",
    [0.0, 5.0, 15.0, 30.0, 45.0, 60.0, 90.0],
)
def test_parity_holds_only_for_collinear(theta_deg: float) -> None:
    """For any deflection angle from collinear, the two diverge.

    At theta=0 the paths cross at the closest-approach point and the
    two agree. At any theta > 0 the paths miss by a positive amount
    and the quadratic returns inf while the label returns the
    closest-approach time.
    """
    theta = math.radians(theta_deg)
    # a moving +x from origin; b moving at angle theta from (2, 0)
    vbx = -math.cos(theta)
    vby = -math.sin(theta)
    df = _df(
        [
            (0, 1, 0.0, 0.0, 1.0, 0.0),
            (0, 2, 2.0, 0.0, vbx, vby),
        ]
    )
    label_ttc = _label_min_ttc(df)
    ssm_ttc = _ssm_ttc(
        pos_a=(0.0, 0.0),
        vel_a=(1.0, 0.0),
        pos_b=(2.0, 0.0),
        vel_b=(vbx, vby),
        min_distance=0.0,
    )
    if theta_deg == 0.0:
        assert math.isfinite(label_ttc)
        assert math.isfinite(ssm_ttc)
        assert label_ttc == pytest.approx(ssm_ttc, rel=1e-9)
    else:
        # Label still returns a positive closest-approach time.
        assert math.isfinite(label_ttc)
        assert label_ttc > 0
        # SSM returns inf: courses do not intersect within radius 0.
        assert math.isinf(ssm_ttc)


def test_label_ttc_is_closest_approach_formula() -> None:
    """Direct check: for a pair with positive closing rate the label's
    TTC equals |rel_pos| / |closing_speed| is not correct; it equals
    -dot(rel_pos, rel_vel) / |rel_vel|^2. Verify that arithmetic."""
    r = np.array([3.0, 4.0])  # separation vector
    v = np.array([-1.0, 0.0])  # relative velocity
    df = _df(
        [
            (0, 1, 0.0, 0.0, 0.0, 0.0),
            (0, 2, r[0], r[1], v[0], v[1]),
        ]
    )
    label_ttc = _label_min_ttc(df)
    expected = -float(np.dot(r, v)) / float(np.dot(v, v))
    assert label_ttc == pytest.approx(expected, abs=1e-12)


def test_ssm_is_collision_radius_quadratic() -> None:
    """Direct check: SSM with radius r and a collinear approach gives
    (|rel_pos| - r) / closing_speed."""
    rel_pos = np.array([10.0, 0.0])
    rel_vel = np.array([-2.0, 0.0])  # closing speed 2 m/s
    radius = 1.5
    ttc = _ssm_ttc(
        pos_a=(0.0, 0.0),
        vel_a=(0.0, 0.0),
        pos_b=rel_pos,
        vel_b=rel_vel,
        min_distance=radius,
    )
    expected = (float(np.linalg.norm(rel_pos)) - radius) / 2.0
    assert ttc == pytest.approx(expected, abs=1e-12)
