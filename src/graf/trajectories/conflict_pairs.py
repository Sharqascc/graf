from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass(slots=True)
class ConflictPair:
    track_id_a: str
    track_id_b: str
    frame_indices: List[int]
    min_distance: float
    min_distance_frame: int
    avg_relative_speed: float


def find_nearby_pairs(
    df: pd.DataFrame,
    frame_col: str = "frame_idx",
    track_col: str = "track_id",
    x_col: str = "x_m",
    y_col: str = "y_m",
    distance_threshold: float = 15.0,
) -> Dict[int, List[Tuple[str, str]]]:
    nearby_pairs: Dict[int, List[Tuple[str, str]]] = {}

    for frame, group in df.groupby(frame_col):
        if len(group) < 2:
            continue

        tracks = group[track_col].astype(str).values
        positions = group[[x_col, y_col]].to_numpy(dtype=float)

        frame_pairs: List[Tuple[str, str]] = []
        for i, j in combinations(range(len(tracks)), 2):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < distance_threshold:
                pair = tuple(sorted((str(tracks[i]), str(tracks[j]))))
                frame_pairs.append(pair)

        if frame_pairs:
            nearby_pairs[int(frame)] = frame_pairs

    return nearby_pairs


def compute_conflict_pairs(
    df: pd.DataFrame,
    frame_col: str = "frame_idx",
    track_col: str = "track_id",
    x_col: str = "x_m",
    y_col: str = "y_m",
    speed_col: str = "speed_mps",
    min_interaction_frames: int = 3,
    distance_threshold: float = 15.0,
) -> List[ConflictPair]:
    nearby = find_nearby_pairs(
        df=df,
        frame_col=frame_col,
        track_col=track_col,
        x_col=x_col,
        y_col=y_col,
        distance_threshold=distance_threshold,
    )

    pair_frames: Dict[Tuple[str, str], List[int]] = {}
    for frame, pairs in nearby.items():
        for pair in pairs:
            pair_frames.setdefault(pair, []).append(frame)

    conflict_pairs: List[ConflictPair] = []

    for (track_a, track_b), frames in pair_frames.items():
        frames_sorted = sorted(frames)
        if len(frames_sorted) < min_interaction_frames:
            continue

        pair_df = df[df[track_col].astype(str).isin([track_a, track_b])]

        min_dist = float("inf")
        min_dist_frame = -1
        rel_speeds: List[float] = []

        for frame in frames_sorted:
            frame_data = pair_df[pair_df[frame_col] == frame]
            if len(frame_data) != 2:
                continue

            row_a = frame_data[frame_data[track_col].astype(str) == track_a].iloc[0]
            row_b = frame_data[frame_data[track_col].astype(str) == track_b].iloc[0]

            pos_a = np.array([row_a[x_col], row_a[y_col]], dtype=float)
            pos_b = np.array([row_b[x_col], row_b[y_col]], dtype=float)
            dist = float(np.linalg.norm(pos_a - pos_b))

            if dist < min_dist:
                min_dist = dist
                min_dist_frame = int(frame)

            if speed_col in frame_data.columns:
                speed_a = float(row_a.get(speed_col, 0.0))
                speed_b = float(row_b.get(speed_col, 0.0))
                if not np.isnan(speed_a) and not np.isnan(speed_b):
                    rel_speeds.append(abs(speed_a - speed_b))

        avg_rel_speed = float(np.mean(rel_speeds)) if rel_speeds else 0.0

        conflict_pairs.append(
            ConflictPair(
                track_id_a=track_a,
                track_id_b=track_b,
                frame_indices=frames_sorted,
                min_distance=min_dist,
                min_distance_frame=min_dist_frame,
                avg_relative_speed=avg_rel_speed,
            )
        )

    return conflict_pairs


def filter_meaningful_conflicts(
    conflict_pairs: List[ConflictPair],
    max_min_distance: float = 5.0,
    min_avg_speed: float = 0.0,
) -> List[ConflictPair]:
    return [
        pair for pair in conflict_pairs
        if pair.min_distance <= max_min_distance and pair.avg_relative_speed >= min_avg_speed
    ]
