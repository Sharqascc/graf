#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from graf.ssm.pet import PETCalculator
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.conflict_pairs import compute_conflict_pairs, filter_meaningful_conflicts
from graf.trajectories.kinematics import compute_kinematics, smooth_positions


def compute_pairwise_ttc_events(
    df_kin: pd.DataFrame,
    candidate_pairs,
    video_id: str,
    ttc_threshold_s: float = 3.0,
):
    records = []

    for pair in candidate_pairs:
        track_a = pair.track_id_a
        track_b = pair.track_id_b

        df_a = df_kin[df_kin["track_id"].astype(str) == str(track_a)]
        df_b = df_kin[df_kin["track_id"].astype(str) == str(track_b)]

        merged = df_a.merge(
            df_b,
            on="frame_idx",
            suffixes=("_a", "_b"),
            how="inner",
        )

        if merged.empty:
            continue

        best_ttc = float("inf")
        best_row = None
        best_result = None

        for _, row in merged.iterrows():
            pos_a = np.array([row["x_m_a"], row["y_m_a"]], dtype=float)
            pos_b = np.array([row["x_m_b"], row["y_m_b"]], dtype=float)
            vel_a = np.array([
                0.0 if pd.isna(row["vx_mps_a"]) else row["vx_mps_a"],
                0.0 if pd.isna(row["vy_mps_a"]) else row["vy_mps_a"],
            ], dtype=float)
            vel_b = np.array([
                0.0 if pd.isna(row["vx_mps_b"]) else row["vx_mps_b"],
                0.0 if pd.isna(row["vy_mps_b"]) else row["vy_mps_b"],
            ], dtype=float)

            result = compute_ttc_constant_velocity(
                pos1=pos_a,
                vel1=vel_a,
                pos2=pos_b,
                vel2=vel_b,
                min_distance=1.5,
            )

            if np.isfinite(result.ttc_seconds) and result.ttc_seconds < best_ttc:
                best_ttc = result.ttc_seconds
                best_row = row
                best_result = result

        if best_result is None or best_row is None:
            continue

        if best_ttc <= ttc_threshold_s:
            records.append({
                "video_id": video_id,
                "event_id": f"TTC_{track_a}_{track_b}_{int(best_row['frame_idx'])}",
                "metric_name": "TTC",
                "track_id_a": str(track_a),
                "track_id_b": str(track_b),
                "start_frame": int(best_row["frame_idx"]),
                "end_frame": int(best_row["frame_idx"]),
                "min_value": float(best_ttc),
                "threshold": float(ttc_threshold_s),
                "severity": "critical",
                "metadata": {
                    "status": best_result.status,
                    "collision_point": best_result.collision_point,
                    "pair_min_distance_m": pair.min_distance,
                    "pair_avg_relative_speed": pair.avg_relative_speed,
                },
            })

    return records


def main():
    parser = argparse.ArgumentParser(description="GRAF SSM Computation Engine")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to world-coordinate trajectories CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save SSM event CSV")
    parser.add_argument("--fps", type=float, default=30.0, help="Video FPS")
    parser.add_argument("--smooth_window", type=int, default=5, help="Centered rolling smoothing window")
    parser.add_argument("--pair_distance_m", type=float, default=12.0, help="Candidate interaction distance")
    parser.add_argument("--pair_frames", type=int, default=3, help="Minimum interaction frames")
    parser.add_argument("--pet_zone_m", type=float, default=2.5, help="Conflict zone radius for PET")
    parser.add_argument("--pet_threshold_s", type=float, default=5.0, help="Critical PET threshold")
    parser.add_argument("--ttc_threshold_s", type=float, default=3.0, help="Critical TTC threshold")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    required = {"track_id", "frame_idx", "x_m", "y_m"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV missing columns: {sorted(missing)}")

    df = smooth_positions(df, window=args.smooth_window)
    df_kin = compute_kinematics(df, fps=args.fps)

    candidate_pairs = compute_conflict_pairs(
        df_kin,
        min_interaction_frames=args.pair_frames,
        distance_threshold=args.pair_distance_m,
    )
    candidate_pairs = filter_meaningful_conflicts(
        candidate_pairs,
        max_min_distance=args.pair_distance_m,
        min_avg_speed=0.0,
    )

    video_id = str(df_kin["video_id"].iloc[0]) if "video_id" in df_kin.columns else "unknown_video"

    pet_engine = PETCalculator(
        proximity_threshold_m=args.pet_zone_m,
        critical_threshold_s=args.pet_threshold_s,
    )

    records = []

    for pair in candidate_pairs:
        df_a = df_kin[df_kin["track_id"].astype(str) == str(pair.track_id_a)]
        df_b = df_kin[df_kin["track_id"].astype(str) == str(pair.track_id_b)]
        pet_event = pet_engine.compute_pair_pet(df_a, df_b, video_id=video_id)
        if pet_event is not None:
            records.append(pet_event.to_dict())

    records.extend(
        compute_pairwise_ttc_events(
            df_kin=df_kin,
            candidate_pairs=candidate_pairs,
            video_id=video_id,
            ttc_threshold_s=args.ttc_threshold_s,
        )
    )

    out_df = pd.DataFrame(records)
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)

    if out_df.empty:
        out_df = pd.DataFrame(columns=[
            "video_id", "event_id", "metric_name", "track_id_a", "track_id_b",
            "start_frame", "end_frame", "min_value", "threshold", "severity", "metadata"
        ])

    out_df.to_csv(args.output_csv, index=False)
    print(f"Saved {len(out_df)} SSM events to {args.output_csv}")


if __name__ == "__main__":
    main()
