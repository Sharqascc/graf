"""DRAC (Deceleration Rate to Avoid Crash) — planned, not yet implemented.

DRAC measures the constant deceleration an actor would need to apply to
avoid a collision with another actor, given current kinematics. The FHWA
Surrogate Safety Assessment Model defines it as:

    DRAC = v_rel**2 / (2 * d)

where ``v_rel`` is the relative speed at braking onset and ``d`` is the gap
distance between the two actors at that moment. A common critical threshold
is 3.35 m/s^2 (the config default in ``configs/ssm/drac.yaml``).

This module is a placeholder kept so downstream imports and the README's
SSM feature list have a documented landing spot. The eventual public API
will mirror :mod:`graf.ssm.ttc`:

    @dataclass
    class DRACResult:
        drac_mps2: float
        is_critical: bool
        status: str

    def compute_drac_constant_velocity(
        pos1, vel1, pos2, vel2, *, gap_override=None
    ) -> DRACResult: ...

Nothing is exported from here yet — see the tests in
``tests/test_ssm_stubs.py`` for the current (stub) behavior.
"""

from __future__ import annotations

__all__: list[str] = []


class DRACResult:  # pragma: no cover
    """Placeholder dataclass — see module docstring for the planned API."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise NotImplementedError(
            "DRAC is planned but not implemented. "
            "See src/graf/ssm/drac.py for the intended API."
        )


def compute_drac_constant_velocity(*args: object, **kwargs: object) -> object:
    """Placeholder — see module docstring for the planned API."""
    raise NotImplementedError(
        "DRAC is planned but not implemented. "
        "See src/graf/ssm/drac.py for the intended API."
    )
