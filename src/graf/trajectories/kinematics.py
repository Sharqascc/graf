from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_seconds(df: pd.DataFrame, fps: float, frame_col: str = "frame_idx") -> pd.DataFrame:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")
    out = df.copy()
    out["t_sec"] = out[frame_col].astype(float) / float(fps)
    return out


def smooth_positions(
    df: pd.DataFrame,
    track_id_col: str = "track_id",
    x_col: str = "x_m",
    y_col: str = "y_m",
    window: int = 3,
) -> pd.DataFrame:
    if window < 1:
        raise ValueError("window must be >= 1")

    out = df.copy()
    out[x_col] = (
        out.groupby(track_id_col, sort=False)[x_col]
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )
    out[y_col] = (
        out.groupby(track_id_col, sort=False)[y_col]
        .transform(lambda s: s.rolling(window, center=True, min_periods=1).mean())
    )
    return out


def compute_kinematics(
    df: pd.DataFrame,
    track_id_col: str = "track_id",
    frame_col: str = "frame_idx",
    x_col: str = "x_m",
    y_col: str = "y_m",
    fps: float = 30.0,
) -> pd.DataFrame:
    if fps <= 0:
        raise ValueError(f"fps must be > 0, got {fps}")

    required = {track_id_col, frame_col, x_col, y_col}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    out = df.copy().sort_values([track_id_col, frame_col]).reset_index(drop=True)
    out["t_sec"] = out[frame_col].astype(float) / float(fps)

    grouped = out.groupby(track_id_col, sort=False)

    out["dx"] = grouped[x_col].diff()
    out["dy"] = grouped[y_col].diff()
    out["dt"] = grouped["t_sec"].diff()

    out["distance_m"] = np.sqrt(out["dx"] ** 2 + out["dy"] ** 2)
    out["speed_mps"] = out["distance_m"] / out["dt"]
    out["heading_rad"] = np.arctan2(out["dy"], out["dx"])
    out["vx_mps"] = out["dx"] / out["dt"]
    out["vy_mps"] = out["dy"] / out["dt"]
    out["accel_mps2"] = grouped["speed_mps"].diff() / out["dt"]

    for col in ["speed_mps", "vx_mps", "vy_mps", "accel_mps2"]:
        out[col] = out[col].replace([np.inf, -np.inf], np.nan)

    return out
