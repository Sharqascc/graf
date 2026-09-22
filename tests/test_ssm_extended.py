import numpy as np
import pandas as pd
import pytest

from graf.ssm.pet import (
    PETCalculator,
    PETResult,
    compute_pet_from_conflict_zone,
)
from graf.ssm.ttc import TTCResult, compute_ttc_constant_velocity


# ---------------------------------------------------------------------------
# PETResult properties
# ---------------------------------------------------------------------------
def test_pet_result_is_critical():
    assert PETResult(0.5).is_critical is True
    assert PETResult(3.0).is_critical is True
    assert PETResult(3.1).is_critical is False
    assert PETResult(-1.0).is_critical is False
    assert PETResult(float("inf")).is_critical is False


def test_pet_result_severity():
    assert PETResult(float("inf")).severity == 0.0
    assert PETResult(-1.0).severity == 1.0
    assert PETResult(0.0).severity == 1.0
    assert PETResult(2.5).severity == pytest.approx(0.5)
    assert PETResult(5.0).severity == 0.0


# ---------------------------------------------------------------------------
# compute_pet_from_conflict_zone
# ---------------------------------------------------------------------------
def test_compute_pet_empty_trajectory():
    result = compute_pet_from_conflict_zone(
        traj1=np.empty((0, 2)),
        traj2=np.array([[0.0, 0.0]]),
        timestamps1=np.array([]),
        timestamps2=np.array([0.0]),
        conflict_zone_center=np.array([0.0, 0.0]),
    )
    assert result.pet_seconds == float("inf")
    assert result.status == "empty_trajectory"


def test_compute_pet_agent1_then_agent2():
    # agent1 enters/exits early, agent2 enters later
    traj1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    traj2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    time1 = np.array([0.0, 1.0])
    time2 = np.array([2.0, 3.0])
    center = np.array([0.5, 0.0])
    result = compute_pet_from_conflict_zone(
        traj1, traj2, time1, time2, center, zone_radius=1.0
    )
    # agent1 exits at t=1.0, agent2 enters at t=2.0 => PET=1.0
    assert result.pet_seconds == pytest.approx(1.0)
    assert result.enters_first == "agent1"
    assert result.status == "agent1_then_agent2"


def test_compute_pet_zone_overlap():
    traj1 = np.array([[0.0, 0.0], [1.0, 0.0]])
    traj2 = np.array([[0.0, 0.0], [1.0, 0.0]])
    time1 = np.array([0.0, 1.0])
    time2 = np.array([0.5, 1.5])
    center = np.array([0.5, 0.0])
    result = compute_pet_from_conflict_zone(
        traj1, traj2, time1, time2, center, zone_radius=1.0
    )
    assert result.pet_seconds == 0.0
    assert result.status == "zone_overlap"


# ---------------------------------------------------------------------------
# PETCalculator
# ---------------------------------------------------------------------------
def test_pet_calculator_missing_columns():
    calc = PETCalculator()
    df_a = pd.DataFrame({"track_id": ["a"], "frame_idx": [0]})
    df_b = pd.DataFrame({"track_id": ["b"], "frame_idx": [0]})
    with pytest.raises(ValueError, match="Missing columns for PET"):
        calc.compute_pair_pet(df_a, df_b, video_id="v1")


def test_pet_calculator_short_trajectory_returns_none():
    calc = PETCalculator()
    df_a = pd.DataFrame(
        {
            "track_id": ["a"],
            "frame_idx": [0],
            "t_sec": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "track_id": ["b"],
            "frame_idx": [0],
            "t_sec": [0.0],
            "x_m": [0.0],
            "y_m": [0.0],
        }
    )
    assert calc.compute_pair_pet(df_a, df_b, video_id="v1") is None


def test_pet_calculator_success():
    calc = PETCalculator(proximity_threshold_m=5.0, critical_threshold_s=5.0)
    df_a = pd.DataFrame(
        {
            "track_id": ["a"] * 3,
            "frame_idx": [0, 1, 2],
            "t_sec": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "track_id": ["b"] * 3,
            "frame_idx": [0, 1, 2],
            "t_sec": [2.0, 3.0, 4.0],
            "x_m": [2.0, 1.0, 0.0],
            "y_m": [0.0, 0.0, 0.0],
        }
    )
    event = calc.compute_pair_pet(df_a, df_b, video_id="v1")
    assert event is not None
    assert event.metric_name == "PET"
    assert event.video_id == "v1"
    assert event.severity in {"critical", "non_critical"}
    assert "conflict_x" in event.metadata


# ---------------------------------------------------------------------------
# TTCResult properties
# ---------------------------------------------------------------------------
def test_ttc_result_is_critical():
    assert TTCResult(0.5).is_critical is True
    assert TTCResult(3.0).is_critical is True
    assert TTCResult(3.1).is_critical is False
    assert TTCResult(0.0).is_critical is False
    assert TTCResult(float("inf")).is_critical is False


def test_ttc_result_severity():
    assert TTCResult(float("inf")).severity == 0.0
    assert TTCResult(-1.0).severity == 1.0
    assert TTCResult(0.0).severity == 1.0
    assert TTCResult(2.5).severity == pytest.approx(0.5)
    assert TTCResult(5.0).severity == 0.0


# ---------------------------------------------------------------------------
# compute_ttc_constant_velocity
# ---------------------------------------------------------------------------
def test_compute_ttc_zero_relative_speed():
    # Actors 5 m apart (well outside the default min_distance=1.5) with
    # identical velocities — no closing at all.
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    assert result.ttc_seconds == float("inf")
    assert result.status == "zero_relative_speed"
    assert not result.is_approaching


def test_compute_ttc_diverging():
    # Actors 5 m apart, moving away from each other (not stationary).
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([5.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert result.ttc_seconds == float("inf")
    assert result.status == "diverging_or_parallel"
    assert not result.is_approaching


def test_compute_ttc_no_collision():
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 10.0]),
        np.array([-1.0, 0.0]),
    )
    assert result.ttc_seconds == float("inf")
    assert result.is_approaching is True
    assert result.status.startswith("no_collision_min_sep")


def test_compute_ttc_collision_predicted():
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([10.0, 0.0]),
        np.array([-1.0, 0.0]),
    )
    assert np.isfinite(result.ttc_seconds)
    assert result.ttc_seconds > 0
    assert result.status == "collision_predicted"
    assert result.is_approaching is True
    assert result.collision_point is not None


def test_compute_ttc_already_in_collision_pair():
    """Two actors inside min_distance at t=0 give TTC=0."""
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
        min_distance=1.5,
    )
    assert result.ttc_seconds == 0.0
    assert result.status == "already_in_collision"
    assert result.is_approaching is True


def test_compute_ttc_already_in_collision_boundary():
    """Exactly at min_distance counts as in collision (<=)."""
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.5, 0.0]),
        np.array([0.0, 0.0]),
        min_distance=1.5,
    )
    assert result.status == "already_in_collision"
    assert result.ttc_seconds == 0.0


def test_compute_ttc_just_outside_min_distance():
    """Just outside min_distance should NOT trigger already_in_collision."""
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.51, 0.0]),
        np.array([0.0, 0.0]),
        min_distance=1.5,
    )
    assert result.status != "already_in_collision"
    assert result.ttc_seconds != 0.0


# ------------------------------------------------------------------
# Mutation-survivor tests (from docs/mutation_testing_2026_10.md)
# ------------------------------------------------------------------


def test_ttc_returns_entry_root_not_exit_root() -> None:
    """The returned TTC is time-to-*first* contact (entry root), not the
    time the actors separate again (exit root).

    Actor 1 at origin, actor 2 approaching head-on from +10 at -1 m/s.
    With min_distance=2.0, they enter the collision radius at t=8 and
    exit at t=12. The function must return 8, not 12.
    """
    from graf.ssm.ttc import compute_ttc_constant_velocity

    pos1 = np.array([0.0, 0.0])
    vel1 = np.array([0.0, 0.0])
    pos2 = np.array([10.0, 0.0])
    vel2 = np.array([-1.0, 0.0])
    r = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=2.0)
    assert np.isfinite(r.ttc_seconds)
    assert abs(r.ttc_seconds - 8.0) < 0.5, (
        f"expected entry time ~8.0, got {r.ttc_seconds:.3f} (exit time would be ~12.0)"
    )


def test_ttc_severity_at_horizon_interior() -> None:
    """Severity at ttc=4.5 must be 0.1 (1 - 4.5/5), not 0.

    Catches the mutant that moves the horizon from 5.0 to 4.0: under that
    mutation, severity at 4.5 would be 0.
    """
    from graf.ssm.ttc import TTCResult

    r = TTCResult(4.5)
    assert abs(r.severity - 0.1) < 1e-9, (
        f"severity at ttc=4.5 should be 0.1, got {r.severity:.6f} "
        "(horizon may have moved from 5.0)"
    )
    # and at 3.5
    assert abs(TTCResult(3.5).severity - 0.3) < 1e-9
    # and at 4.9
    assert abs(TTCResult(4.9).severity - 0.02) < 1e-9


def test_ttc_tangent_rotation_invariant() -> None:
    """A near-tangent pair stays on the same branch under rotation.

    The scale-aware tangent tolerance exists so the discriminant sign is
    stable under rotation. Without it, the same geometry rotated by a
    tiny angle flips between collision_predicted and no_collision.
    """
    from graf.ssm.ttc import compute_ttc_constant_velocity

    # Head-on with a tiny perpendicular offset that nearly tangents the
    # collision radius.
    for angle in (0.0, 1e-7, 1e-5, 1e-3):
        c, s = np.cos(angle), np.sin(angle)
        pos1 = np.array([0.0, 0.0])
        vel1 = np.array([0.0, 0.0])
        pos2 = np.array([10.0 * c, 10.0 * s])
        vel2 = np.array([-1.0 * c, -1.0 * s])
        r = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=2.0)
        # All angles in this set should be on the same branch.
        assert np.isfinite(r.ttc_seconds) or "no_collision" in r.status, (
            f"angle={angle}: status={r.status}, ttc={r.ttc_seconds}"
        )


def test_ttc_non_finite_inputs_return_sentinel() -> None:
    """Non-finite inputs return a non-critical inf-TTC sentinel, not a raise.

    Matches the contract established by tests/test_input_boundaries.py:
    the pipeline runs over real tracking data where NaN positions occur,
    so a malformed input must degrade, not crash.
    """
    from graf.ssm.ttc import compute_ttc_constant_velocity

    r = compute_ttc_constant_velocity(
        np.array([np.nan, 0.0]),
        np.zeros(2),
        np.array([1.0, 0.0]),
        np.zeros(2),
    )
    assert r.status == "invalid_input"
    assert not r.is_critical
    assert not np.isfinite(r.ttc_seconds)

    r = compute_ttc_constant_velocity(
        np.array([np.inf, 0.0]),
        np.zeros(2),
        np.array([1.0, 0.0]),
        np.zeros(2),
    )
    assert r.status == "invalid_input"
    assert not r.is_critical


def test_ttc_negative_min_distance_clamps_to_zero() -> None:
    """Negative min_distance is nonsensical; treat as zero-radius, not crash."""
    from graf.ssm.ttc import compute_ttc_constant_velocity

    pos1, vel1 = np.zeros(2), np.zeros(2)
    pos2, vel2 = np.array([10.0, 0.0]), np.array([-5.0, 0.0])
    r = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=-1.0)
    r_zero = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=0.0)
    assert r.status == r_zero.status
    assert abs(r.ttc_seconds - r_zero.ttc_seconds) < 1e-9
