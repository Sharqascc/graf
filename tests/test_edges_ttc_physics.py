"""Physics tests for compute_ttc_simple in graf.graph.edges.

Item 15: before the fix, compute_ttc_simple returned the *exit* root
of the collision quadratic when two actors were already inside the
collision radius at t=0 and separating. That value reads as
"collision in t seconds" to any downstream consumer, which is the
opposite of the truth.

The correct behavior matches graf.ssm.ttc.compute_ttc_constant_velocity:
when the pair is already in contact, TTC is 0, regardless of relative
motion.
"""

from __future__ import annotations

import numpy as np
import pytest

from graf.graph.edges import compute_ttc_simple
from graf.ssm.ttc import compute_ttc_constant_velocity


def _ttc(x1, y1, vx1, vy1, x2, y2, vx2, vy2, radius=1.5):
    return compute_ttc_simple(
        x1,
        y1,
        vx1,
        vy1,
        x2,
        y2,
        vx2,
        vy2,
        collision_radius=radius,
    )


def test_already_in_collision_moving_apart_returns_zero():
    """The bug case: inside radius, separating.

    Before the fix this returned 0.25 (the exit root), which reads as
    "collision in 0.25 s". Truth: already in contact.
    """
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0)
    assert ttc == 0.0


def test_already_in_collision_moving_toward_returns_zero():
    """Inside collision radius, approaching: TTC is 0, not negative."""
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, -2.0, 0.0)
    assert ttc == 0.0


def test_already_in_collision_stationary_returns_zero():
    """Inside collision radius, no relative motion: TTC is 0."""
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
    assert ttc == 0.0


def test_exactly_at_collision_boundary_returns_zero():
    """Distance exactly equal to collision_radius: already touching."""
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0, 0.0)
    assert ttc == 0.0


def test_just_outside_collision_radius_closing_returns_positive():
    """Distance 1.6 m, closing at 2 m/s, radius 1.5 m: TTC = 0.05 s."""
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 1.6, 0.0, -2.0, 0.0)
    assert ttc == pytest.approx(0.05, abs=1e-9)


def test_just_outside_collision_radius_diverging_returns_max_ttc():
    """Distance 1.6 m, separating: never reaches contact."""
    ttc = compute_ttc_simple(
        0.0,
        0.0,
        0.0,
        0.0,
        1.6,
        0.0,
        2.0,
        0.0,
        collision_radius=1.5,
        max_ttc=10.0,
    )
    assert ttc == 10.0


def test_far_apart_head_on_returns_finite():
    """Control: 10 m apart, closing 10 m/s, radius 1.5 m.

    Contact at t = 0.85 s. The new already-in-collision check must not
    affect this case.
    """
    ttc = _ttc(0.0, 0.0, 0.0, 0.0, 10.0, 0.0, -10.0, 0.0)
    assert ttc == pytest.approx(0.85, abs=1e-9)


def test_agrees_with_ssm_module_on_already_in_collision():
    """Cross-check against the module this one is supposed to mirror."""
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([1.0, 0.0])
    vel1 = np.array([0.0, 0.0])
    vel2 = np.array([2.0, 0.0])

    ssm_result = compute_ttc_constant_velocity(
        pos1,
        vel1,
        pos2,
        vel2,
        min_distance=1.5,
    )
    assert ssm_result.ttc_seconds == 0.0
    assert ssm_result.status == "already_in_collision"

    simple_ttc = _ttc(
        pos1[0],
        pos1[1],
        vel1[0],
        vel1[1],
        pos2[0],
        pos2[1],
        vel2[0],
        vel2[1],
    )
    assert simple_ttc == 0.0


@pytest.mark.parametrize(
    "radius",
    [0.5, 1.5, 3.0],
    ids=["small", "default", "large"],
)
def test_zero_inside_any_collision_radius(radius):
    """Inside radius r means TTC = 0 for any r."""
    ttc = _ttc(
        0.0,
        0.0,
        0.0,
        0.0,
        radius * 0.5,
        0.0,
        2.0,
        0.0,
        radius=radius,
    )
    assert ttc == 0.0
