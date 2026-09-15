"""Compute per-frame SSM values (TTC, DRAC) from world-space trajectories.

Reads a trajectories file in one of three shapes:

* Nested JSON from ``scripts/build_trajectories.py`` — a list of
  ``{track_id, class_name, frames: [{frame_id, x, y}]}`` objects.
* Flat JSON array — a list of rows with columns ``frame_idx``,
  ``track_id``, ``x_m``, ``y_m`` and optional ``vx_mps``/``vy_mps``.
* JSONL — one JSON object per line, same columns as the flat array.

For every pair of tracks within ``--distance-threshold`` on a frame, emits
one row per metric (TTC and DRAC) with the columns::

    video_id, frame_idx, track_id_a, track_id_b, metric_name, value

This is the format ``scripts/mine_ssm_events.py`` consumes. Non-finite
values (e.g. TTC = +inf for diverging actors) are dropped from the output
because they cannot become events; the event-mining semantics already
treat non-finite as "not crossing".

PET is intentionally not computed here. PET is defined against a specific
conflict-zone location rather than a pair of actors, and picking a
per-pair derivation (midpoint, closest approach, ...) is a modelling
decision that depends on the annotated site geometry. It will be added
alongside the real-data pipeline.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from graf.ssm.drac import compute_drac_constant_velocity
from graf.ssm.event_mining import load_ssm_thresholds, mine_events
from graf.ssm.ttc import compute_ttc_constant_velocity
from graf.trajectories.conflict_pairs import find_nearby_pairs
from graf.trajectories.kinematics import compute_kinematics
from graf.utils.io import ensure_dir, write_json, write_jsonl

_VALUE_COLUMNS = (
    "video_id",
    "frame_idx",
    "track_id_a",
    "track_id_b",
    "metric_name",
    "value",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-frame TTC and DRAC for nearby track pairs."
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        default="outputs/trajectories/trajectories.json",
        help="Input trajectories file (nested JSON, flat JSON, or JSONL).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/ssm_values",
        help="Directory for ssm_values.jsonl and summary.json.",
    )
    parser.add_argument(
        "--video-id",
        type=str,
        default="unknown",
        help="Video identifier to stamp on rows (falls back to any per-row value).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=23.98,
        help="Frames per second, used when deriving velocities.",
    )
    parser.add_argument(
        "--distance-threshold",
        type=float,
        default=15.0,
        help="Maximum pair distance to compute SSM (metres).",
    )
    parser.add_argument(
        "--collision-radius",
        type=float,
        default=1.5,
        help="Sum of actor radii defining contact (metres).",
    )
    parser.add_argument(
        "--mine-events",
        action="store_true",
        help="After computing values, also run event mining.",
    )
    parser.add_argument(
        "--mine-config-dir",
        type=str,
        default="configs/ssm",
        help="Directory of YAML threshold configs (used with --mine-events).",
    )
    parser.add_argument(
        "--mine-outdir",
        type=str,
        default="outputs/ssm_events",
        help="Output directory for mined events (used with --mine-events).",
    )
    return parser.parse_args(argv)


def _flatten_nested(payload: list[dict[str, Any]]) -> pd.DataFrame:
    """Flatten the nested build_trajectories.py output into per-frame rows."""
    rows: list[dict[str, Any]] = []
    for track in payload:
        track_id = track.get("track_id")
        class_name = track.get("class_name", "unknown")
        video_id = track.get("video_id")
        for frame in track.get("frames", []):
            rows.append(
                {
                    "video_id": video_id,
                    "frame_idx": int(frame.get("frame_id", frame.get("frame_idx"))),
                    "track_id": track_id,
                    "class_name": class_name,
                    "x_m": float(frame["x"]),
                    "y_m": float(frame["y"]),
                }
            )
    return pd.DataFrame(rows)


def load_trajectories(path: str | Path) -> pd.DataFrame:
    """Load trajectories into a flat DataFrame with frame_idx/track_id/x_m/y_m.

    Handles nested JSON (build_trajectories.py), flat JSON arrays, and
    JSONL. Rows are not re-ordered; kinematics will sort internally.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame()

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)

    if isinstance(payload, dict) and "trajectories" in payload:
        payload = payload["trajectories"]
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()

    first = payload[0]
    if isinstance(first, dict) and "frames" in first:
        return _flatten_nested(payload)
    return pd.DataFrame(payload)


def _normalize_flat(df: pd.DataFrame, video_id: str) -> pd.DataFrame:
    """Ensure required columns, harmonise x/y aliases, stamp video_id."""
    if df.empty:
        return df
    df = df.copy()
    # Accept x/y as well as x_m/y_m
    if "x_m" not in df.columns and "x" in df.columns:
        df["x_m"] = df["x"]
    if "y_m" not in df.columns and "y" in df.columns:
        df["y_m"] = df["y"]
    if "video_id" not in df.columns:
        df["video_id"] = video_id
    else:
        df["video_id"] = df["video_id"].fillna(video_id)
    return df


def _has_velocities(df: pd.DataFrame) -> bool:
    return {"vx_mps", "vy_mps"}.issubset(df.columns)


def compute_values(
    df: pd.DataFrame,
    video_id: str,
    fps: float,
    distance_threshold: float,
    collision_radius: float,
) -> list[dict[str, Any]]:
    """Compute per-frame TTC and DRAC rows for nearby track pairs."""
    if df.empty:
        return []
    if not _has_velocities(df):
        df = compute_kinematics(df, fps=fps)
    if "vx_mps" not in df.columns or "vy_mps" not in df.columns:
        raise ValueError(
            "Trajectories lack velocities and compute_kinematics did not "
            "produce vx_mps/vy_mps."
        )

    # Drop rows with NaN velocities (typically the first frame of each track).
    df = df.dropna(subset=["vx_mps", "vy_mps"]).reset_index(drop=True)
    if df.empty:
        return []

    nearby = find_nearby_pairs(df, distance_threshold=distance_threshold)
    if not nearby:
        return []

    by_frame: dict[int, dict[str, dict[str, Any]]] = {}
    for frame, group in df.groupby("frame_idx"):
        by_frame[int(frame)] = {
            str(row["track_id"]): row for _, row in group.iterrows()
        }

    rows: list[dict[str, Any]] = []
    for frame, pairs in nearby.items():
        actors = by_frame.get(int(frame), {})
        for ta, tb in pairs:
            ra = actors.get(ta)
            rb = actors.get(tb)
            if ra is None or rb is None:
                continue

            pos_a = np.array([ra["x_m"], ra["y_m"]], dtype=float)
            vel_a = np.array([ra["vx_mps"], ra["vy_mps"]], dtype=float)
            pos_b = np.array([rb["x_m"], rb["y_m"]], dtype=float)
            vel_b = np.array([rb["vx_mps"], rb["vy_mps"]], dtype=float)

            ttc_res = compute_ttc_constant_velocity(
                pos_a, vel_a, pos_b, vel_b, min_distance=collision_radius
            )
            drac_res = compute_drac_constant_velocity(
                pos_a, vel_a, pos_b, vel_b, collision_radius=collision_radius
            )

            for metric, value in (
                ("TTC", ttc_res.ttc_seconds),
                ("DRAC", drac_res.drac_mps2),
            ):
                if not np.isfinite(value):
                    continue
                rows.append(
                    {
                        "video_id": str(ra.get("video_id", video_id)),
                        "frame_idx": int(frame),
                        "track_id_a": ta,
                        "track_id_b": tb,
                        "metric_name": metric,
                        "value": float(value),
                    }
                )
    return rows


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = ensure_dir(args.outdir)

    df = load_trajectories(args.traj_path)
    df = _normalize_flat(df, video_id=args.video_id)

    rows = compute_values(
        df,
        video_id=args.video_id,
        fps=args.fps,
        distance_threshold=args.distance_threshold,
        collision_radius=args.collision_radius,
    )

    values_path = outdir / "ssm_values.jsonl"
    write_jsonl(values_path, rows)

    summary: dict[str, Any] = {
        "num_rows": len(rows),
        "num_input_trajectory_rows": int(len(df)),
        "metrics": sorted({r["metric_name"] for r in rows}),
        "fps": args.fps,
        "distance_threshold": args.distance_threshold,
        "collision_radius": args.collision_radius,
        "input": str(Path(args.traj_path)),
    }
    write_json(outdir / "summary.json", summary)
    print(f"Wrote {len(rows)} SSM value rows to {values_path}")

    if args.mine_events:
        thresholds = load_ssm_thresholds(args.mine_config_dir)
        if not thresholds:
            print("No thresholds found — skipping event mining.")
            return 0
        values_df = pd.DataFrame(rows, columns=list(_VALUE_COLUMNS))
        events = mine_events(values_df, thresholds=thresholds)
        mine_outdir = ensure_dir(args.mine_outdir)
        write_jsonl(
            mine_outdir / "ssm_events.jsonl",
            [e.to_dict() for e in events],
        )
        write_json(
            mine_outdir / "summary.json",
            {
                "num_events": len(events),
                "thresholds": thresholds,
                "num_input_rows": len(rows),
            },
        )
        print(f"Wrote {len(events)} SSM events to {mine_outdir / 'ssm_events.jsonl'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
