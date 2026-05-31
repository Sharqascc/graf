from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from graf.data.schema import SSMEventRecord


@dataclass(slots=True)
class PETResult:
    pet_seconds: float
    conflict_point: Optional[Tuple[float, float]] = None
    enters_first: Optional[str] = None
    status: str = "uncomputed"

    @property
    def is_critical(self) -> bool:
        return np.isfinite(self.pet_seconds) and 0 <= self.pet_seconds <= 3.0

    @property
    def severity(self) -> float:
        if not np.isfinite(self.pet_seconds):
            return 0.0
        if self.pet_seconds <= 0:
            return 1.0
        if self.pet_seconds >= 5.0:
            return 0.0
        return float(1.0 - (self.pet_seconds / 5.0))


def compute_pet_from_conflict_zone(
    traj1: np.ndarray,
    traj2: np.ndarray,
    timestamps1: np.ndarray,
    timestamps2: np.ndarray,
    conflict_zone_center: np.ndarray,
    zone_radius: float = 2.0,
) -> PETResult:
    traj1 = np.asarray(traj1, dtype=float)
    traj2 = np.asarray(traj2, dtype=float)
    timestamps1 = np.asarray(timestamps1, dtype=float)
    timestamps2 = np.asarray(timestamps2, dtype=float)
    center = np.asarray(conflict_zone_center, dtype=float)

    if len(traj1) == 0 or len(traj2) == 0:
        return PETResult(float("inf"), None, None, "empty_trajectory")

    dist1 = np.linalg.norm(traj1 - center, axis=1)
    dist2 = np.linalg.norm(traj2 - center, axis=1)

    in_zone1 = dist1 <= zone_radius
    in_zone2 = dist2 <= zone_radius

    if not np.any(in_zone1) or not np.any(in_zone2):
        return PETResult(float("inf"), tuple(center.tolist()), None, "one_or_both_never_enter_zone")

    enter1_idx = int(np.argmax(in_zone1))
    enter2_idx = int(np.argmax(in_zone2))
    exit1_idx = int(len(in_zone1) - 1 - np.argmax(in_zone1[::-1]))
    exit2_idx = int(len(in_zone2) - 1 - np.argmax(in_zone2[::-1]))

    enter1_time = float(timestamps1[enter1_idx])
    enter2_time = float(timestamps2[enter2_idx])
    exit1_time = float(timestamps1[exit1_idx])
    exit2_time = float(timestamps2[exit2_idx])

    if exit1_time <= enter2_time:
        pet = enter2_time - exit1_time
        enters_first = "agent1"
        status = "agent1_then_agent2"
    elif exit2_time <= enter1_time:
        pet = enter1_time - exit2_time
        enters_first = "agent2"
        status = "agent2_then_agent1"
    else:
        pet = 0.0
        enters_first = None
        status = "zone_overlap"

    return PETResult(
        pet_seconds=float(pet),
        conflict_point=tuple(center.tolist()),
        enters_first=enters_first,
        status=status,
    )


class PETCalculator:
    def __init__(self, proximity_threshold_m: float = 2.0, critical_threshold_s: float = 5.0):
        self.proximity_threshold_m = float(proximity_threshold_m)
        self.critical_threshold_s = float(critical_threshold_s)

    def compute_pair_pet(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        video_id: str,
    ) -> Optional[SSMEventRecord]:
        required = {"track_id", "frame_idx", "t_sec", "x_m", "y_m"}
        missing_a = required.difference(df_a.columns)
        missing_b = required.difference(df_b.columns)
        if missing_a or missing_b:
            raise ValueError(f"Missing columns for PET: A={sorted(missing_a)}, B={sorted(missing_b)}")

        if len(df_a) < 2 or len(df_b) < 2:
            return None

        df_a = df_a.sort_values("frame_idx").reset_index(drop=True)
        df_b = df_b.sort_values("frame_idx").reset_index(drop=True)

        track_id_a = str(df_a["track_id"].iloc[0])
        track_id_b = str(df_b["track_id"].iloc[0])

        traj_a = df_a[["x_m", "y_m"]].to_numpy(dtype=float)
        traj_b = df_b[["x_m", "y_m"]].to_numpy(dtype=float)
        time_a = df_a["t_sec"].to_numpy(dtype=float)
        time_b = df_b["t_sec"].to_numpy(dtype=float)

        dists = np.linalg.norm(traj_a[:, None, :] - traj_b[None, :, :], axis=-1)
        min_idx = np.unravel_index(np.argmin(dists), dists.shape)
        min_dist = float(dists[min_idx])

        if min_dist > self.proximity_threshold_m:
            return None

        conflict_center = ((traj_a[min_idx[0]] + traj_b[min_idx[1]]) / 2.0).astype(float)
        pet_result = compute_pet_from_conflict_zone(
            traj1=traj_a,
            traj2=traj_b,
            timestamps1=time_a,
            timestamps2=time_b,
            conflict_zone_center=conflict_center,
            zone_radius=self.proximity_threshold_m,
        )

        if not np.isfinite(pet_result.pet_seconds):
            return None

        event = SSMEventRecord(
            video_id=video_id,
            event_id=f"PET_{track_id_a}_{track_id_b}",
            metric_name="PET",
            track_id_a=track_id_a,
            track_id_b=track_id_b,
            start_frame=int(min(df_a["frame_idx"].min(), df_b["frame_idx"].min())),
            end_frame=int(max(df_a["frame_idx"].max(), df_b["frame_idx"].max())),
            min_value=float(pet_result.pet_seconds),
            threshold=self.critical_threshold_s,
            severity="critical" if pet_result.pet_seconds <= self.critical_threshold_s else "non_critical",
            metadata={
                "conflict_x": float(conflict_center[0]),
                "conflict_y": float(conflict_center[1]),
                "min_pair_distance_m": float(min_dist),
                "pet_status": pet_result.status,
                "enters_first": pet_result.enters_first,
            },
        )
        return event
