import math

import numpy as np
import pandas as pd
import pytest

from graf.ssm.pet import PETCalculator, compute_pet_from_conflict_zone
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.conflict_pairs import find_nearby_pairs


def test_ttc_head_on():
    pos1 = np.array([0.0, 0.0])
    pos2 = np.array([10.0, 0.0])
    vel1 = np.array([5.0, 0.0])
    vel2 = np.array([-5.0, 0.0])

    result = compute_ttc_constant_velocity(pos1, vel1, pos2, vel2, min_distance=0.0)
    assert abs(result.ttc_seconds - 1.0) < 1e-6
    assert result.is_approaching
    assert result.is_critical


def test_pet_conflict_zone_sequential():
    traj1 = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])
    traj2 = np.array([[2.0, 1.0], [2.0, 0.6], [2.0, 0.3], [2.0, 0.0]])
    t1 = np.array([0.0, 1.0, 2.0, 3.0])
    t2 = np.array([3.2, 4.2, 5.2, 6.2])

    result = compute_pet_from_conflict_zone(
        traj1=traj1,
        traj2=traj2,
        timestamps1=t1,
        timestamps2=t2,
        conflict_zone_center=np.array([2.0, 0.0]),
        zone_radius=0.25,
    )
    assert result.pet_seconds >= 0.0
    assert np.isfinite(result.pet_seconds)


def test_find_nearby_pairs():
    df = pd.DataFrame(
        {
            "frame_idx": [0, 0, 0, 1, 1, 1],
            "track_id": ["a", "b", "c", "a", "b", "c"],
            "x_m": [0, 2, 100, 0, 2, 100],
            "y_m": [0, 2, 100, 0, 2, 100],
        }
    )

    pairs = find_nearby_pairs(df, distance_threshold=5.0)
    assert 0 in pairs
    assert ("a", "b") in pairs[0]
    assert ("a", "c") not in pairs[0]
    assert 1 in pairs
    assert ("a", "b") in pairs[1]


def test_pet_calculator_event():
    df_a = pd.DataFrame(
        {
            "track_id": ["a", "a", "a"],
            "frame_idx": [0, 1, 2],
            "t_sec": [0.0, 1.0, 2.0],
            "x_m": [0.0, 1.0, 2.0],
            "y_m": [0.0, 0.0, 0.0],
        }
    )
    df_b = pd.DataFrame(
        {
            "track_id": ["b", "b", "b"],
            "frame_idx": [3, 4, 5],
            "t_sec": [2.2, 3.2, 4.2],
            "x_m": [2.0, 2.0, 2.0],
            "y_m": [1.0, 0.2, 0.0],
        }
    )

    calc = PETCalculator(proximity_threshold_m=0.5)
    event = calc.compute_pair_pet(df_a, df_b, video_id="vid_test")
    assert event is not None
    assert event.metric_name == "PET"
    assert event.track_id_a == "a"
    assert event.track_id_b == "b"


# ------------------------------------------------------------------
# DRAC
# ------------------------------------------------------------------

from graf.ssm.drac import (
    CRITICAL_DRAC_MPS2,
    DRACResult,
    compute_drac_constant_velocity,
)


def test_drac_head_on():
    """Two cars closing at 2 m/s with 8.5 m gap."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(1.0, 0.0),
        pos2=(10.0, 0.0),
        vel2=(-1.0, 0.0),
    )
    assert r.status == "computed"
    assert r.closing_speed_mps == pytest.approx(2.0)
    assert r.gap_m == pytest.approx(8.5)
    assert r.drac_mps2 == pytest.approx(4.0 / (2.0 * 8.5))
    assert r.is_critical is False


def test_drac_critical_when_gap_small():
    """5 m/s closing on a 2 m effective gap exceeds the 3.35 threshold."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(2.5, 0.0),
        pos2=(3.5, 0.0),
        vel2=(-2.5, 0.0),
    )
    assert r.gap_m == pytest.approx(2.0)
    assert r.drac_mps2 == pytest.approx(25.0 / 4.0)
    assert r.is_critical is True


def test_drac_diverging_is_zero():
    """Away-moving actors -> 0 DRAC with the diverging status."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(-1.0, 0.0),
        pos2=(10.0, 0.0),
        vel2=(1.0, 0.0),
    )
    assert r.status == "diverging_or_parallel"
    assert r.drac_mps2 == 0.0
    assert r.is_critical is False


def test_drac_already_in_collision():
    """Overlapping collision circles -> +inf and already_in_collision."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(1.0, 0.0),
        pos2=(1.0, 0.0),
        vel2=(-1.0, 0.0),
    )
    assert r.status == "already_in_collision"
    assert math.isinf(r.drac_mps2)
    assert r.severity == 1.0


def test_drac_zero_relative_speed():
    """Equal velocities -> 0 DRAC and zero_relative_speed status."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(1.0, 0.0),
        pos2=(10.0, 0.0),
        vel2=(1.0, 0.0),
    )
    assert r.status == "zero_relative_speed"
    assert r.drac_mps2 == 0.0


def test_drac_gap_override():
    """gap_override replaces the position-derived distance."""
    r = compute_drac_constant_velocity(
        pos1=(0.0, 0.0),
        vel1=(1.0, 0.0),
        pos2=(1000.0, 0.0),
        vel2=(-1.0, 0.0),
        gap_override=5.5,
    )
    assert r.gap_m == pytest.approx(4.0)
    assert r.drac_mps2 == pytest.approx(4.0 / 8.0)


def test_drac_result_dataclass_defaults():
    r = DRACResult(0.0)
    assert r.status == "uncomputed"
    assert r.closing_speed_mps == 0.0
    assert r.gap_m == 0.0


def test_drac_critical_threshold_constant():
    assert CRITICAL_DRAC_MPS2 == 3.35
