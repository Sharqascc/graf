from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class TTCResult:
    ttc_seconds: float
    collision_point: tuple[float, float] | None = None
    is_approaching: bool = False
    status: str = "uncomputed"

    @property
    def is_critical(self) -> bool:
        return bool(np.isfinite(self.ttc_seconds) and 0 < self.ttc_seconds <= 3.0)

    @property
    def severity(self) -> float:
        if not np.isfinite(self.ttc_seconds):
            return 0.0
        if self.ttc_seconds <= 0:
            return 1.0
        if self.ttc_seconds >= 5.0:
            return 0.0
        return float(1.0 - (self.ttc_seconds / 5.0))


def compute_ttc_constant_velocity(
    pos1: np.ndarray,
    vel1: np.ndarray,
    pos2: np.ndarray,
    vel2: np.ndarray,
    min_distance: float = 1.5,
) -> TTCResult:
    pos1 = np.asarray(pos1, dtype=float)
    vel1 = np.asarray(vel1, dtype=float)
    pos2 = np.asarray(pos2, dtype=float)
    vel2 = np.asarray(vel2, dtype=float)

    # Non-finite inputs have no meaningful TTC. Return a sentinel rather
    # than raise: the tests in test_input_boundaries.py establish the
    # contract that malformed input must not crash the pipeline (it runs
    # over real tracking data where NaN positions occur). A sentinel that
    # is never critical is the right shape for that caller.
    if (
        not np.all(np.isfinite(pos1))
        or not np.all(np.isfinite(pos2))
        or not np.all(np.isfinite(vel1))
        or not np.all(np.isfinite(vel2))
    ):
        return TTCResult(float("inf"), None, False, "invalid_input")

    # min_distance == 0.0 is a valid "point collision" (the actors collide
    # when they reach the same point). Negative is nonsensical but we
    # treat it as a zero-radius collision rather than crash, for the same
    # reason as above: bad input from real data must degrade, not raise.
    if min_distance < 0:
        min_distance = 0.0

    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1

    # If the actors are already inside the collision radius at t=0, TTC is
    # zero. Without this check the quadratic below returns the *exit*
    # root (when they separate), which is not time-to-collision.
    #
    # Snap the comparison with a scale-aware epsilon: distance exactly
    # equal to min_distance can round to ±1e-15 under rotation, flipping
    # this <= nondeterministically. See test_ttc_rotation_invariant.
    _d = float(np.linalg.norm(rel_pos))
    _tol = 1e-9 * max(1.0, _d, min_distance)
    if _d <= min_distance + _tol:
        return TTCResult(0.0, None, True, "already_in_collision")

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-12:
        return TTCResult(float("inf"), None, False, "zero_relative_speed")

    closing_rate = -float(np.dot(rel_pos, rel_vel))
    # Perpendicular or near-perpendicular pairs have closing_rate ~ 0; the
    # sign of dot(rel_pos, rel_vel) is then float64 noise and flips under
    # rotation. Reject on a tolerance proportional to the magnitude scale
    # so the status is rotation-invariant.
    noise_floor = (
        1e-9 * float(np.linalg.norm(rel_pos)) * float(np.sqrt(rel_speed_sq)) + 1e-12
    )
    if closing_rate <= noise_floor:
        return TTCResult(float("inf"), None, False, "diverging_or_parallel")

    a = rel_speed_sq
    b = 2.0 * float(np.dot(rel_pos, rel_vel))
    c = float(np.dot(rel_pos, rel_pos) - min_distance**2)

    discriminant = b**2 - 4.0 * a * c
    # Near-zero discriminant is a tangent approach: the two actors just
    # graze the collision radius. Tiny float64 error flips the sign
    # between +0 and -0, which then picks the "collision" vs
    # "no_collision" branch non-deterministically. Snap |disc| within a
    # scale-aware epsilon to zero so the result is rotation-invariant
    # (see tests/test_metamorphic.py::test_ttc_rotation_invariant).
    _eps = 1e-12 * max(1.0, b * b, 4.0 * a * abs(c))
    if discriminant < -_eps:
        t_near = closing_rate / rel_speed_sq
        closest_sep = np.linalg.norm(rel_pos + t_near * rel_vel)
        return TTCResult(
            float("inf"),
            None,
            True,
            f"no_collision_min_sep_{closest_sep:.3f}",
        )
    if discriminant < 0.0:
        discriminant = 0.0

    sqrt_disc = float(np.sqrt(discriminant))
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive_roots = [t for t in roots if t > 0]

    if not positive_roots:
        return TTCResult(float("inf"), None, True, "collision_in_past_or_now")

    ttc = float(min(positive_roots))
    collision_point = tuple((pos1 + vel1 * ttc).tolist())

    return TTCResult(ttc, collision_point, True, "collision_predicted")
