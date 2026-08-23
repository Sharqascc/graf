import pandas as pd
import pytest

from graf.trajectories.conflict_pairs import (
    ConflictPair,
    compute_conflict_pairs,
    filter_meaningful_conflicts,
    find_nearby_pairs,
)


def make_tracks_df(data):
    return pd.DataFrame(data)


def test_find_nearby_pairs_basic():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 1, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "b", "x_m": 100.0, "y_m": 0.0},
        ]
    )
    pairs = find_nearby_pairs(df, distance_threshold=10.0)
    assert 1 in pairs
    assert pairs[1] == [("a", "b")]
    # Frame 2 should be absent because distance 100 > 10
    assert 2 not in pairs


def test_find_nearby_pairs_ignores_single_track():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
        ]
    )
    pairs = find_nearby_pairs(df)
    assert pairs == {}


def test_find_nearby_pairs_sorts_pair():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "b", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 1, "track_id": "a", "x_m": 1.0, "y_m": 0.0},
        ]
    )
    pairs = find_nearby_pairs(df, distance_threshold=10.0)
    assert pairs[1] == [("a", "b")]


def test_compute_conflict_pairs_basic():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "a", "x_m": 0.0, "y_m": 0.0, "speed_mps": 1.0},
            {"frame_idx": 1, "track_id": "b", "x_m": 1.0, "y_m": 0.0, "speed_mps": 2.0},
            {"frame_idx": 2, "track_id": "a", "x_m": 0.0, "y_m": 0.0, "speed_mps": 1.5},
            {"frame_idx": 2, "track_id": "b", "x_m": 0.5, "y_m": 0.0, "speed_mps": 2.5},
            {"frame_idx": 3, "track_id": "a", "x_m": 0.0, "y_m": 0.0, "speed_mps": 2.0},
            {"frame_idx": 3, "track_id": "b", "x_m": 0.2, "y_m": 0.0, "speed_mps": 3.0},
        ]
    )
    conflicts = compute_conflict_pairs(
        df, min_interaction_frames=3, distance_threshold=5.0
    )
    assert len(conflicts) == 1
    c = conflicts[0]
    assert c.track_id_a == "a"
    assert c.track_id_b == "b"
    assert c.frame_indices == [1, 2, 3]
    assert c.min_distance == pytest.approx(0.2)
    assert c.min_distance_frame == 3
    # avg_relative_speed: abs(1-2)=1, abs(1.5-2.5)=1, abs(2-3)=1 => 1.0
    assert c.avg_relative_speed == pytest.approx(1.0)


def test_compute_conflict_pairs_min_frames_not_met():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 1, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
        ]
    )
    conflicts = compute_conflict_pairs(
        df, min_interaction_frames=3, distance_threshold=5.0
    )
    assert conflicts == []


def test_compute_conflict_pairs_no_speed_col():
    df = pd.DataFrame(
        [
            {"frame_idx": 1, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 1, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 2, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
            {"frame_idx": 3, "track_id": "a", "x_m": 0.0, "y_m": 0.0},
            {"frame_idx": 3, "track_id": "b", "x_m": 1.0, "y_m": 0.0},
        ]
    )
    conflicts = compute_conflict_pairs(
        df, min_interaction_frames=3, distance_threshold=5.0
    )
    assert len(conflicts) == 1
    assert conflicts[0].avg_relative_speed == 0.0


def test_filter_meaningful_conflicts():
    pairs = [
        ConflictPair(
            "a",
            "b",
            [1, 2],
            min_distance=3.0,
            min_distance_frame=1,
            avg_relative_speed=2.0,
        ),
        ConflictPair(
            "c",
            "d",
            [1, 2],
            min_distance=10.0,
            min_distance_frame=1,
            avg_relative_speed=5.0,
        ),
        ConflictPair(
            "e",
            "f",
            [1, 2],
            min_distance=1.0,
            min_distance_frame=1,
            avg_relative_speed=0.0,
        ),
    ]
    filtered = filter_meaningful_conflicts(
        pairs, max_min_distance=5.0, min_avg_speed=0.5
    )
    assert len(filtered) == 1
    assert filtered[0].track_id_a == "a"

    # No speed threshold
    filtered2 = filter_meaningful_conflicts(
        pairs, max_min_distance=5.0, min_avg_speed=0.0
    )
    assert len(filtered2) == 2
