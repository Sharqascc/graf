from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(slots=True)
class TTCResult:
    ttc_seconds: float
    collision_point: Optional[Tuple[float, float]] = None
    is_approaching: bool = False
    status: str = "uncomputed"

    @property
    def is_critical(self) -> bool:
        return np.isfinite(self.ttc_seconds) and 0 < self.ttc_seconds <= 3.0

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

    rel_pos = pos2 - pos1
    rel_vel = vel2 - vel1

    rel_speed_sq = float(np.dot(rel_vel, rel_vel))
    if rel_speed_sq < 1e-12:
        return TTCResult(float("inf"), None, False, "zero_relative_speed")

    closing_rate = -float(np.dot(rel_pos, rel_vel))
    if closing_rate <= 0:
        return TTCResult(float("inf"), None, False, "diverging_or_parallel")

    a = rel_speed_sq
    b = 2.0 * float(np.dot(rel_pos, rel_vel))
    c = float(np.dot(rel_pos, rel_pos) - min_distance ** 2)

    discriminant = b ** 2 - 4.0 * a * c
    if discriminant < 0:
        t_near = closing_rate / rel_speed_sq
        closest_sep = np.linalg.norm(rel_pos + t_near * rel_vel)
        return TTCResult(
            float("inf"),
            None,
            True,
            f"no_collision_min_sep_{closest_sep:.3f}",
        )

    sqrt_disc = float(np.sqrt(discriminant))
    roots = [(-b - sqrt_disc) / (2.0 * a), (-b + sqrt_disc) / (2.0 * a)]
    positive_roots = [t for t in roots if t > 0]

    if not positive_roots:
        return TTCResult(float("inf"), None, True, "collision_in_past_or_now")

    ttc = float(min(positive_roots))
    collision_point = tuple((pos1 + vel1 * ttc).tolist())

    return TTCResult(ttc, collision_point, True, "collision_predicted")
