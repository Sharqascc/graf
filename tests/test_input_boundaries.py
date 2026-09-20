"""Input boundary tests: malformed data must fail loudly, not silently.

CV and trajectory pipelines produce malformed data — a missing frame,
a NaN from a division, an inverted bbox, a zero fps — and the worst
failure mode is silent: the pipeline returns a plausible number that
is wrong. These tests assert that each public primitive either raises
a clear exception or returns a value that is obviously degenerate
(NaN/inf), never a plausible-looking wrong number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from graf.calibration.homography import project_points
from graf.ssm.drac import compute_drac_constant_velocity
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.kinematics import compute_kinematics

# ------------------------------------------------------------------
# compute_kinematics
# ------------------------------------------------------------------


def _tracks(n=3, x_start=0.0):
    return pd.DataFrame(
        {
            "track_id": [1] * n,
            "frame_idx": list(range(n)),
            "x_m": [x_start + i for i in range(n)],
            "y_m": [0.0] * n,
        }
    )


def test_kinematics_negative_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        compute_kinematics(_tracks(), fps=-1.0)


def test_kinematics_zero_fps_raises():
    with pytest.raises(ValueError, match="fps"):
        compute_kinematics(_tracks(), fps=0.0)


def test_kinematics_missing_x_column_raises():
    df = _tracks().drop(columns=["x_m"])
    with pytest.raises(ValueError, match="Missing"):
        compute_kinematics(df)


def test_kinematics_single_row_produces_nan_velocities():
    out = compute_kinematics(_tracks(n=1), fps=30.0)
    assert len(out) == 1
    # First row of any track has no previous row; velocity is undefined.
    assert pd.isna(out["vx_mps"].iloc[0])
    assert pd.isna(out["vy_mps"].iloc[0])


def test_kinematics_nan_coords_propagate_as_nan():
    df = _tracks(n=3)
    df.loc[1, "x_m"] = float("nan")
    out = compute_kinematics(df, fps=30.0)
    # Row 1 and row 2 velocities should be NaN — not a finite garbage value.
    assert pd.isna(out["vx_mps"].iloc[1])
    assert pd.isna(out["vx_mps"].iloc[2])


def test_kinematics_inf_coords_replaced_with_nan():
    df = _tracks(n=3)
    df.loc[1, "x_m"] = float("inf")
    out = compute_kinematics(df, fps=30.0)
    # compute_kinematics replaces +-inf with NaN explicitly
    assert not np.isinf(out["vx_mps"]).any()
    assert not np.isinf(out["speed_mps"]).any()


def test_kinematics_unsorted_frames_still_produces_valid_velocity():
    df = _tracks(n=3)
    df = df.iloc[::-1].reset_index(drop=True)  # reversed frame order
    out = compute_kinematics(df, fps=30.0)
    # Kinematics sorts before differencing, so velocity is +1 m/frame
    valid = out["vx_mps"].dropna()
    assert (valid > 0).all()


# ------------------------------------------------------------------
# project_points
# ------------------------------------------------------------------


def test_project_points_identity_matrix():
    H = np.eye(3)
    out = project_points(H, [(1.0, 2.0), (3.0, 4.0)])
    np.testing.assert_allclose(out, [[1.0, 2.0], [3.0, 4.0]])


def test_project_points_zero_scale_raises():
    # Third row all zero => homogeneous denominator is 0 for any point.
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    with pytest.raises(ZeroDivisionError):
        project_points(H, [(1.0, 2.0)])


def test_project_points_near_zero_scale_raises():
    # Third row tiny but non-zero: isclose catches it before division.
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1e-15]])
    with pytest.raises(ZeroDivisionError):
        project_points(H, [(1.0, 2.0)])


def test_project_points_valid_matrix_does_not_raise():
    # Control: a normal non-degenerate H must NOT raise.
    H = np.array([[2.0, 0.0, 5.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]])
    out = project_points(H, [(1.0, 2.0)])
    np.testing.assert_allclose(out, [[7.0, 9.0]])


# ------------------------------------------------------------------
# compute_ttc_constant_velocity
# ------------------------------------------------------------------


def test_ttc_nan_position_does_not_return_plausible_number():
    r = compute_ttc_constant_velocity(
        pos1=np.array([np.nan, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([10.0, 0.0]),
        vel2=np.array([-1.0, 0.0]),
    )
    # Either NaN or inf, but never a finite plausible-looking number.
    assert not np.isfinite(r.ttc_seconds) or r.ttc_seconds == 0.0


def test_ttc_inf_position_handled():
    r = compute_ttc_constant_velocity(
        pos1=np.array([np.inf, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([10.0, 0.0]),
        vel2=np.array([-1.0, 0.0]),
    )
    # Any status is acceptable as long as no crash and result is a
    # recognised string.
    assert isinstance(r.status, str)


def test_ttc_large_coordinates_no_overflow():
    r = compute_ttc_constant_velocity(
        pos1=np.array([0.0, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([1e12, 0.0]),
        vel2=np.array([-1.0, 0.0]),
    )
    # Finite answer expected; no overflow.
    assert np.isfinite(r.ttc_seconds)


def test_ttc_zero_min_distance_does_not_crash():
    r = compute_ttc_constant_velocity(
        pos1=np.array([0.0, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([10.0, 0.0]),
        vel2=np.array([-1.0, 0.0]),
        min_distance=0.0,
    )
    assert isinstance(r.status, str)


# ------------------------------------------------------------------
# compute_drac_constant_velocity
# ------------------------------------------------------------------


def test_drac_nan_position_does_not_return_plausible_number():
    r = compute_drac_constant_velocity(
        pos1=np.array([np.nan, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([10.0, 0.0]),
        vel2=np.array([-1.0, 0.0]),
    )
    # Either NaN or 0.0 (diverging); never a plausible positive number.
    assert not np.isfinite(r.drac_mps2) or r.drac_mps2 == 0.0


def test_drac_large_coordinates_no_overflow():
    r = compute_drac_constant_velocity(
        pos1=np.array([0.0, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([1e6, 0.0]),
        vel2=np.array([-1.0, 0.0]),
    )
    assert np.isfinite(r.drac_mps2)


def test_drac_zero_collision_radius_does_not_crash():
    r = compute_drac_constant_velocity(
        pos1=np.array([0.0, 0.0]),
        vel1=np.array([1.0, 0.0]),
        pos2=np.array([10.0, 0.0]),
        vel2=np.array([-1.0, 0.0]),
        collision_radius=0.0,
    )
    assert isinstance(r.status, str)


# ------------------------------------------------------------------
# build_trajectories — malformed inputs
# ------------------------------------------------------------------


def test_build_trajectories_empty_rows_returns_empty():
    from scripts.build_trajectories import build_trajectories

    assert build_trajectories([], homography=None, pixels_per_meter=None) == []


def test_build_trajectories_missing_video_id_defaults():
    from scripts.build_trajectories import build_trajectories

    rows = [
        {
            "frame_idx": 0,
            "track_id": 1,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "class_name": "car",
        }
    ]
    payload = build_trajectories(rows, homography=None, pixels_per_meter=None)
    assert payload[0]["video_id"] == "unknown"


def test_build_trajectories_malformed_bbox_raises():
    from scripts.build_trajectories import build_trajectories

    rows = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 1,
            "bbox_xyxy": [0.0, 0.0, 10.0],
            "class_name": "car",
        }
    ]
    with pytest.raises(ValueError):
        build_trajectories(rows, homography=None, pixels_per_meter=None)


def test_build_trajectories_negative_ppm_raises():
    from scripts.build_trajectories import build_trajectories

    rows = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 1,
            "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
            "class_name": "car",
        }
    ]
    with pytest.raises(ValueError, match="> 0"):
        build_trajectories(rows, homography=None, pixels_per_meter=-1.0)
