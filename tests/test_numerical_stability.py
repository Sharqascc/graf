"""Numerical stability of the SSM math under float64 edge cases.

The paper's labels come out of arithmetic on world coordinates that are
themselves the product of a homography, a tracker, and a filter. Every
step is a float64 operation on a value that has already lost precision.
These tests assert that the SSM functions do not amplify that loss into
behavioral flips: a boundary case that is 1e-12 over or under an exact
threshold should give the same answer as the exact threshold.
"""

from __future__ import annotations

import numpy as np
import pytest

from graf.ssm.ttc import TTCResult, compute_ttc_constant_velocity


def test_ttc_survives_near_identical_positions() -> None:
    base = 1e6
    pos1 = np.array([base, base])
    pos2 = np.array([base + 1e-6, base])
    vel1 = np.array([0.0, 0.0])
    vel2 = np.array([-1.0, 0.0])
    r = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=0.0)
    assert isinstance(r.status, str)
    assert np.isfinite(r.ttc_seconds) or np.isinf(r.ttc_seconds)


def test_ttc_does_not_return_nan_at_small_separation() -> None:
    for sep in (1e-3, 1e-6, 1e-9, 1e-10):
        pos1 = np.zeros(2)
        pos2 = np.array([sep, 0.0])
        r = compute_ttc_constant_velocity(
            pos1, np.zeros(2), pos2, np.array([-1.0, 0.0]), min_distance=0.0
        )
        assert not np.isnan(r.ttc_seconds), f"sep={sep}: TTC is NaN"


@pytest.mark.parametrize("scale", [1e-6, 1e-3, 1e0, 1e3, 1e6])
def test_ttc_scale_invariant_under_coordinate_scaling(scale: float) -> None:
    pos1 = np.zeros(2)
    pos2 = np.array([10.0, 0.0])
    vel2 = np.array([-1.0, 0.0])
    r_base = compute_ttc_constant_velocity(
        pos1, np.zeros(2), pos2, vel2, min_distance=0.0
    )
    r_scaled = compute_ttc_constant_velocity(
        pos1 * scale, np.zeros(2), pos2 * scale, vel2 * scale, min_distance=0.0
    )
    assert abs(r_scaled.ttc_seconds - r_base.ttc_seconds) < 1e-6


def test_ttc_accepts_float32_inputs_and_returns_comparable_result() -> None:
    pos1_64 = np.array([0.0, 0.0], dtype=np.float64)
    pos2_64 = np.array([10.0, 0.0], dtype=np.float64)
    vel2_64 = np.array([-1.0, 0.0], dtype=np.float64)
    r64 = compute_ttc_constant_velocity(
        pos1_64, np.zeros(2), pos2_64, vel2_64, min_distance=0.0
    )
    r32 = compute_ttc_constant_velocity(
        pos1_64.astype(np.float32),
        np.zeros(2, dtype=np.float32),
        pos2_64.astype(np.float32),
        vel2_64.astype(np.float32),
        min_distance=0.0,
    )
    assert abs(r32.ttc_seconds - r64.ttc_seconds) < 1e-5


def test_ttc_min_distance_boundary_stable_across_rotation() -> None:
    min_distance = 1.5
    statuses = set()
    for angle in np.linspace(0, 2 * np.pi, 17):
        c, s = np.cos(angle), np.sin(angle)
        pos1 = np.array([0.0, 0.0])
        pos2 = np.array([min_distance * c, min_distance * s])
        r = compute_ttc_constant_velocity(
            pos1, np.zeros(2), pos2, np.zeros(2), min_distance=min_distance
        )
        statuses.add(r.status)
    assert statuses == {"already_in_collision"}


def test_ttc_severity_is_continuous_at_horizon() -> None:
    eps = 1e-9
    s_below = TTCResult(5.0 - eps).severity
    s_at = TTCResult(5.0).severity
    s_above = TTCResult(5.0 + eps).severity
    assert s_at == 0.0
    assert s_above == 0.0
    assert s_below < 1e-7


@pytest.mark.parametrize(
    "pos1,pos2,vel1,vel2",
    [
        (np.array([0.0, 0.0]), np.array([1e6, 1e6]), np.zeros(2), np.zeros(2)),
        (np.array([1e-10, 0.0]), np.array([1e10, 0.0]), np.zeros(2), np.zeros(2)),
        (np.array([0.0, 0.0]), np.array([0.0, 0.0]), np.zeros(2), np.zeros(2)),
        (
            np.zeros(2),
            np.array([1.0, 1.0]),
            np.array([1.0, 1.0]),
            np.array([-1.0, -1.0]),
        ),
    ],
)
def test_ttc_never_returns_nan(pos1, pos2, vel1, vel2) -> None:
    r = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=1.5)
    assert not np.isnan(r.ttc_seconds)
