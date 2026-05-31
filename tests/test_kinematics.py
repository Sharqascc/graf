import numpy as np
import pandas as pd

from graf.trajectories.kinematics import add_time_seconds, compute_kinematics, smooth_positions


def test_add_time_seconds():
    df = pd.DataFrame({"frame_idx": [0, 15, 30]})
    out = add_time_seconds(df, fps=30)
    assert list(out["t_sec"]) == [0.0, 0.5, 1.0]


def test_compute_kinematics_speed():
    df = pd.DataFrame({
        "track_id": ["a", "a", "a"],
        "frame_idx": [0, 1, 2],
        "x_m": [0.0, 1.0, 2.0],
        "y_m": [0.0, 0.0, 0.0],
    })
    out = compute_kinematics(df, fps=1.0)
    assert np.isnan(out.loc[0, "speed_mps"])
    assert out.loc[1, "speed_mps"] == 1.0
    assert out.loc[2, "speed_mps"] == 1.0
    assert out.loc[1, "heading_rad"] == 0.0
    assert out.loc[1, "vx_mps"] == 1.0
    assert out.loc[1, "vy_mps"] == 0.0


def test_smooth_positions():
    df = pd.DataFrame({
        "track_id": ["a", "a", "a"],
        "x_m": [0.0, 10.0, 20.0],
        "y_m": [0.0, 0.0, 0.0],
    })
    out = smooth_positions(df, window=3)
    assert np.isclose(out.loc[1, "x_m"], 10.0)
