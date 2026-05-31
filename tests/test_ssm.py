import numpy as np
import pandas as pd

from graf.ssm.pet import PETCalculator, compute_pet_from_conflict_zone
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.conflict_pairs import find_nearby_pairs, compute_conflict_pairs


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
    df = pd.DataFrame({
        "frame_idx": [0, 0, 0, 1, 1, 1],
        "track_id": ["a", "b", "c", "a", "b", "c"],
        "x_m": [0, 2, 100, 0, 2, 100],
        "y_m": [0, 2, 100, 0, 2, 100],
    })

    pairs = find_nearby_pairs(df, distance_threshold=5.0)
    assert 0 in pairs
    assert ("a", "b") in pairs[0]
    assert ("a", "c") not in pairs[0]
    assert 1 in pairs
    assert ("a", "b") in pairs[1]


def test_pet_calculator_event():
    df_a = pd.DataFrame({
        "track_id": ["a", "a", "a"],
        "frame_idx": [0, 1, 2],
        "t_sec": [0.0, 1.0, 2.0],
        "x_m": [0.0, 1.0, 2.0],
        "y_m": [0.0, 0.0, 0.0],
    })
    df_b = pd.DataFrame({
        "track_id": ["b", "b", "b"],
        "frame_idx": [3, 4, 5],
        "t_sec": [2.2, 3.2, 4.2],
        "x_m": [2.0, 2.0, 2.0],
        "y_m": [1.0, 0.2, 0.0],
    })

    calc = PETCalculator(proximity_threshold_m=0.5)
    event = calc.compute_pair_pet(df_a, df_b, video_id="vid_test")
    assert event is not None
    assert event.metric_name == "PET"
    assert event.track_id_a == "a"
    assert event.track_id_b == "b"
