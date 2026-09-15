"""Property-based tests for GRAF core utilities."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_point = st.tuples(
    st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
    st.floats(-1e4, 1e4, allow_nan=False, allow_infinity=False),
)
_positive_fps = st.floats(1.0, 120.0, allow_nan=False, allow_infinity=False)

homography = pytest.importorskip("graf.calibration.homography")

@pytest.mark.hypothesis
@given(points=st.lists(_point, min_size=1, max_size=50))
@settings(max_examples=50, deadline=None)
def test_project_points_preserves_count(points):
    out = homography.project_points(np.eye(3), points)
    assert out.shape == (len(points), 2)

@pytest.mark.hypothesis
@given(points=st.lists(_point, min_size=1, max_size=20))
@settings(max_examples=30, deadline=None)
def test_identity_homography_is_identity_map(points):
    projected = homography.project_points(np.eye(3), points)
    for (x, y), (px, py) in zip(points, projected, strict=True):
        assert math.isclose(x, px, rel_tol=1e-9, abs_tol=1e-9)
        assert math.isclose(y, py, rel_tol=1e-9, abs_tol=1e-9)

@pytest.mark.hypothesis
@given(
    a=st.floats(0.5, 2.0, allow_nan=False, allow_infinity=False),
    b=st.floats(-1.0, 1.0, allow_nan=False, allow_infinity=False),
    tx=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
    ty=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
)
@settings(max_examples=30, deadline=None)
def test_invert_is_involution(a, b, tx, ty):
    if abs(a * a - b * b) < 1e-6:
        pytest.skip("singular affine matrix")
    H = np.array([[a, b, tx], [b, a, ty], [0.0, 0.0, 1.0]], dtype=np.float64)
    H_back = homography.invert_homography(homography.invert_homography(H))
    np.testing.assert_allclose(H, H_back, rtol=1e-6, atol=1e-6)

kinematics = pytest.importorskip("graf.trajectories.kinematics")

def _make_tracks(n_frames: int, fps: float) -> pd.DataFrame:
    frames = np.arange(n_frames, dtype=float)
    t = frames / fps
    return pd.DataFrame({
        "track_id": np.ones(n_frames, dtype=int),
        "frame_idx": frames,
        "x_m": t,
        "y_m": 0.5 * t,
    })

@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_speed_is_non_negative(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    speeds = out["speed_mps"].dropna()
    assert (speeds >= 0).all()

@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_no_infinities_in_kinematics(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    for col in ("speed_mps", "vx_mps", "vy_mps", "accel_mps2"):
        assert not np.isinf(out[col].to_numpy(dtype=float)).any(), col

@pytest.mark.hypothesis
@given(n_frames=st.integers(2, 200), fps=_positive_fps)
@settings(max_examples=30, deadline=None)
def test_time_seconds_matches_frame_over_fps(n_frames, fps):
    out = kinematics.compute_kinematics(_make_tracks(n_frames, fps), fps=fps)
    expected = out["frame_idx"].astype(float) / fps
    np.testing.assert_allclose(out["t_sec"].to_numpy(), expected.to_numpy())
