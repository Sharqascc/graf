
import numpy as np
import pandas as pd
import pytest

from graf.ssm.pet import (
    PETResult,
    compute_pet_from_conflict_zone,
    PETCalculator,
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
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([0.0, 0.0]),
    )
    assert result.ttc_seconds == float("inf")
    assert result.status == "zero_relative_speed"
    assert not result.is_approaching


def test_compute_ttc_diverging():
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),  # same velocity -> zero relative actually
    )
    # Use different velocities that move apart
    result = compute_ttc_constant_velocity(
        np.array([0.0, 0.0]),
        np.array([-1.0, 0.0]),
        np.array([1.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert result.ttc_seconds == float("inf")
    assert result.status in {"diverging_or_parallel", "zero_relative_speed"}
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
