"""Differential tests: optimized pipeline vs a simple reference.

The reference here is deliberately slow and obvious — it integrates the
relative motion forward in tiny steps and reports the first time two
actors come within the collision radius. The optimized implementation
solves the same problem in closed form via a quadratic. If the two
diverge by more than the integration step size, the closed-form code is
wrong (or the reference is — either way, investigate).

This test suite is intentionally small: the reference is O(horizon/dt)
per call, so we keep examples and dt modest.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from graf.ssm.ttc import compute_ttc_constant_velocity

# Point strategies chosen so the pair is likely to actually collide —
# otherwise almost every example gives inf on both sides and the test
# never exercises the interesting path.
_pos1 = st.tuples(
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
)
_pos2 = st.tuples(
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
    st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False),
)
_vel = st.tuples(
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
    st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
)


def _naive_ttc(p1, v1, p2, v2, min_distance, dt=0.01, horizon=10.0):
    """Forward-integrate relative motion, return first time within min_distance.

    Uses pure-Python scalars so the inner loop is fast enough for pre-commit;
    numpy array overhead dominates at these iteration counts.
    """
    ax, ay = float(p1[0]), float(p1[1])
    bx, by = float(p2[0]), float(p2[1])
    vax, vay = float(v1[0]), float(v1[1])
    vbx, vby = float(v2[0]), float(v2[1])
    md2 = min_distance * min_distance
    n = int(horizon / dt)
    for i in range(n + 1):
        dx = bx - ax
        dy = by - ay
        if dx * dx + dy * dy <= md2:
            return i * dt
        ax += vax * dt
        ay += vay * dt
        bx += vbx * dt
        by += vby * dt
    return float("inf")


@pytest.mark.hypothesis
@given(p1=_pos1, v1=_vel, p2=_pos2, v2=_vel)
@settings(max_examples=15, deadline=None)
def test_ttc_matches_naive_integration(p1, v1, p2, v2):
    min_distance = 1.5
    fast = compute_ttc_constant_velocity(p1, v1, p2, v2, min_distance=min_distance)
    slow = _naive_ttc(p1, v1, p2, v2, min_distance=min_distance)
    dt = 0.01

    fast_finite = np.isfinite(fast.ttc_seconds)
    slow_finite = np.isfinite(slow)

    # If one is finite and the other is not, and the difference is not
    # just a quantization boundary, that is a real bug. We tolerate one
    # such case by checking tolerance below.
    if fast_finite and slow_finite:
        assert abs(fast.ttc_seconds - slow) <= 2 * dt + 1e-6, (
            f"closed-form TTC {fast.ttc_seconds} vs naive {slow}"
        )
    elif not fast_finite and not slow_finite:
        pass  # both agree there is no collision
    else:
        # Agreement cases: fast says already_in_collision (ttc=0) and naive
        # also returns 0; or the finite side is near the horizon (a
        # boundary quantization effect).
        if fast.status == "already_in_collision" and slow == 0.0:
            pass
        else:
            ttc = fast.ttc_seconds if fast_finite else slow
            assert ttc > 8.0, (
                f"disagreement not at horizon: closed-form={fast.ttc_seconds} "
                f"naive={slow} status={fast.status}"
            )


def test_ttc_exact_known_case():
    """Pin down one hand-computed case so the naive reference has a witness."""
    # Actors 10 m apart, closing head-on at 1 m/s each. Closing speed 2 m/s.
    # Collision radius 1.5 m -> contact when distance = 1.5 m -> 8.5 m of gap
    # to close -> 8.5 / 2 = 4.25 s.
    r = compute_ttc_constant_velocity(
        [0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [-1.0, 0.0], min_distance=1.5
    )
    assert r.status == "collision_predicted"
    assert abs(r.ttc_seconds - 4.25) < 1e-9

    naive = _naive_ttc([0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [-1.0, 0.0], 1.5)
    assert abs(naive - 4.25) <= 2 * 0.01 + 1e-6
