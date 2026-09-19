"""DRAC (Deceleration Rate to Avoid Crash).

DRAC measures the constant deceleration an actor would need to apply to
avoid a collision with another actor, given current kinematics. The FHWA
Surrogate Safety Assessment Model defines it as:

    DRAC = v_rel**2 / (2 * d)

where ``v_rel`` is the *closing speed* — the component of relative velocity
along the line connecting the two actors, positive when they are
approaching — and ``d`` is the *effective gap*: the Euclidean separation
minus the collision radius, i.e. the distance to contact rather than the
centre-to-centre distance.

The default critical threshold is 3.35 m/s^2, matching the FHWA SSAM
recommendation and ``configs/ssm/drac.yaml``.

This mirrors :mod:`graf.ssm.ttc`: dataclass result, ``is_critical`` and
``severity`` properties, and status strings.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: FHWA SSAM default: decelerations at or above this rate are critical.
CRITICAL_DRAC_MPS2: float = 3.35

#: Deceleration above which severity saturates to 1.0. Roughly the limit of
#: emergency braking on dry asphalt; higher values are physically implausible
#: and treated as maximally severe.
MAX_SEVERITY_DRAC_MPS2: float = 10.0

__all__ = [
    "CRITICAL_DRAC_MPS2",
    "MAX_SEVERITY_DRAC_MPS2",
    "DRACResult",
    "compute_drac_constant_velocity",
]


@dataclass(slots=True)
class DRACResult:
    drac_mps2: float
    closing_speed_mps: float = 0.0
    gap_m: float = 0.0
    status: str = "uncomputed"

    @property
    def is_critical(self) -> bool:
        return bool(
            np.isfinite(self.drac_mps2) and self.drac_mps2 >= CRITICAL_DRAC_MPS2
        )

    @property
    def severity(self) -> float:
        # Note: unlike TTC/PET, +inf here means "already in collision" and
        # maps to maximum severity, not zero.
        if np.isinf(self.drac_mps2):
            return 1.0
        if not np.isfinite(self.drac_mps2) or self.drac_mps2 <= 0.0:
            return 0.0
        return float(min(self.drac_mps2 / MAX_SEVERITY_DRAC_MPS2, 1.0))


def compute_drac_constant_velocity(
    pos1: np.ndarray,
    vel1: np.ndarray,
    pos2: np.ndarray,
    vel2: np.ndarray,
    collision_radius: float = 1.5,
    gap_override: float | None = None,
) -> DRACResult:
    """Compute DRAC between two actors with constant-velocity kinematics.

    Parameters
    ----------
    pos1, vel1, pos2, vel2
        2D position and velocity of each actor (m and m/s).
    collision_radius
        Sum of actor radii that defines contact (default 1.5 m, matching
        :func:`graf.ssm.ttc.compute_ttc_constant_velocity`).
    gap_override
        If provided, use this value as the raw centre-to-centre gap instead
        of computing it from ``pos1``/``pos2``. Useful when the caller has a
        better gap estimate (e.g. from bounding boxes with occlusion).

    Returns
    -------
    DRACResult
        ``status`` is one of ``"computed"``, ``"zero_relative_speed"``,
        ``"diverging_or_parallel"``, or ``"already_in_collision"``.
    """
    pos1 = np.asarray(pos1, dtype=float)
    vel1 = np.asarray(vel1, dtype=float)
    pos2 = np.asarray(pos2, dtype=float)
    vel2 = np.asarray(vel2, dtype=float)

    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1

    # Direction of the inter-actor line is always derived from the actual
    # positions. Only the *magnitude* of the gap can be overridden, so that
    # callers with a better gap estimate (e.g. bounding boxes under partial
    # occlusion) still get correct closing-speed computation.
    actual_distance = float(np.linalg.norm(rel_pos))
    if gap_override is not None:
        raw_gap = float(gap_override)
    else:
        raw_gap = actual_distance

    gap = raw_gap - float(collision_radius)

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-12:
        return DRACResult(
            drac_mps2=0.0,
            closing_speed_mps=0.0,
            gap_m=max(gap, 0.0),
            status="zero_relative_speed",
        )

    # Already-in-collision is a geometric fact (gap <= 0), not a
    # kinematic one. Deciding it *after* the closing-speed dispatch made
    # the status flip between "diverging_or_parallel" and
    # "already_in_collision" under rotation, because closing_speed for
    # a perpendicular pair is a signed float64 epsilon — sometimes +,
    # sometimes -. Check geometry first so the status is rotation-
    # invariant.
    if actual_distance <= 1e-12 or gap <= 0.0:
        unit = rel_pos / actual_distance if actual_distance > 1e-12 else np.zeros(2)
        closing_speed = -float(np.dot(unit, rel_vel))
        return DRACResult(
            drac_mps2=float("inf"),
            closing_speed_mps=closing_speed,
            gap_m=0.0,
            status="already_in_collision",
        )

    unit = rel_pos / actual_distance
    closing_speed = -float(np.dot(unit, rel_vel))

    # Same noise floor as TTC: perpendicular pairs have closing_speed ~ 0
    # whose sign is float64 noise and flips under rotation.
    noise_floor = 1e-9 * float(np.sqrt(rel_speed_sq)) + 1e-12
    if closing_speed <= noise_floor:
        return DRACResult(
            drac_mps2=0.0,
            closing_speed_mps=closing_speed,
            gap_m=gap,
            status="diverging_or_parallel",
        )

    drac = (closing_speed * closing_speed) / (2.0 * gap)

    return DRACResult(
        drac_mps2=float(drac),
        closing_speed_mps=closing_speed,
        gap_m=float(gap),
        status="computed",
    )
