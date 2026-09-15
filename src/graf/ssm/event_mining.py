"""Event mining from SSM outputs.

Turns per-frame SSM values (TTC, PET, DRAC) into discrete, mappable safety
events: contiguous windows where a given metric crosses its critical
threshold, attributed to a pair of tracks and a video. Output rows follow
the SSMEventRecord schema already used by graf.ssm.pet.

Direction of the threshold check depends on the metric:

* TTC and PET are times — smaller is more dangerous, so a value is
  critical when it falls at or below the threshold.
* DRAC is a deceleration magnitude — larger is more dangerous, so a value
  is critical when it rises at or above the threshold.

Unknown metric names default to the "below" direction. The table
_METRIC_DIRECTION is the single source of truth for this mapping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from graf.data.schema import SSMEventRecord

__all__ = ["mine_events"]

_REQUIRED_COLUMNS: tuple[str, ...] = (
    "video_id",
    "frame_idx",
    "track_id_a",
    "track_id_b",
    "metric_name",
    "value",
)

_GROUP_KEYS: tuple[str, ...] = (
    "video_id",
    "track_id_a",
    "track_id_b",
    "metric_name",
)

_METRIC_DIRECTION: dict[str, str] = {
    "TTC": "below",
    "PET": "below",
    "DRAC": "above",
}


def _crosses(value: float, threshold: float, direction: str) -> bool:
    """True iff value is on the critical side of threshold.

    Non-finite values never count as crossing.
    """
    if not np.isfinite(value):
        return False
    if direction == "below":
        return value <= threshold
    return value >= threshold


def _emit_event(
    events: list[SSMEventRecord],
    video_id: object,
    track_a: object,
    track_b: object,
    metric: object,
    threshold: float,
    direction: str,
    run_frames: np.ndarray,
    run_values: np.ndarray,
) -> None:
    """Construct one SSMEventRecord and append it to events."""
    start_frame = int(run_frames[0])
    end_frame = int(run_frames[-1])
    extreme = float(run_values.min() if direction == "below" else run_values.max())

    events.append(
        SSMEventRecord(
            video_id=str(video_id),
            event_id=f"{metric}_{track_a}_{track_b}_{start_frame}",
            metric_name=str(metric),
            track_id_a=str(track_a),
            track_id_b=str(track_b),
            start_frame=start_frame,
            end_frame=end_frame,
            min_value=extreme,
            threshold=float(threshold),
            severity="critical",
            metadata={
                "metric_direction": direction,
                "num_frames": int(len(run_frames)),
                "mean_value": float(run_values.mean()),
            },
        )
    )


def mine_events(
    ssm_frame_values: pd.DataFrame,
    thresholds: dict[str, float],
    min_duration_frames: int = 2,
    max_frame_gap: int = 1,
) -> list[SSMEventRecord]:
    """Mine discrete SSM events from a per-frame metric stream.

    Each row of ssm_frame_values is one SSM value for one pair of tracks at
    one frame. mine_events groups rows by (video_id, track_id_a, track_id_b,
    metric_name), marks rows whose value crosses the corresponding
    thresholds[metric_name] in the metric's critical direction (below for
    TTC/PET, above for DRAC), and extracts contiguous runs whose length
    meets min_duration_frames.

    Two rows belong to the same run if their frame_idx differ by at most
    max_frame_gap. The default (1) means strictly consecutive frames; set
    to 2 when the SSM was computed every other frame, and so on.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in ssm_frame_values.columns]
    if missing:
        raise ValueError(f"ssm_frame_values missing required columns: {missing}")

    if min_duration_frames < 1:
        raise ValueError(f"min_duration_frames must be >= 1, got {min_duration_frames}")
    if max_frame_gap < 1:
        raise ValueError(f"max_frame_gap must be >= 1, got {max_frame_gap}")

    if ssm_frame_values.empty or not thresholds:
        return []

    df = ssm_frame_values[ssm_frame_values["metric_name"].isin(thresholds)].copy()
    if df.empty:
        return []

    df = df.sort_values([*_GROUP_KEYS, "frame_idx"]).reset_index(drop=True)

    events: list[SSMEventRecord] = []

    for keys, group in df.groupby(list(_GROUP_KEYS), sort=False):
        _video_id, _ta, _tb, _metric = keys
        threshold = float(thresholds[_metric])
        direction = _METRIC_DIRECTION.get(str(_metric), "below")

        frames = group["frame_idx"].to_numpy()
        values = group["value"].to_numpy(dtype=float)
        crosses = np.fromiter(
            (_crosses(float(v), threshold, direction) for v in values),
            dtype=bool,
            count=len(values),
        )

        runs: list[tuple[int, int]] = []
        run_start: int | None = None
        for i in range(len(frames)):
            hit = bool(crosses[i])
            if not hit:
                if run_start is not None:
                    runs.append((run_start, i - 1))
                    run_start = None
                continue
            if run_start is None:
                run_start = i
                continue
            if frames[i] - frames[i - 1] > max_frame_gap:
                runs.append((run_start, i - 1))
                run_start = i
        if run_start is not None:
            runs.append((run_start, len(frames) - 1))

        for start_i, end_i in runs:
            n = end_i - start_i + 1
            if n < min_duration_frames:
                continue
            _emit_event(
                events,
                _video_id,
                _ta,
                _tb,
                _metric,
                threshold,
                direction,
                frames[start_i : end_i + 1],
                values[start_i : end_i + 1],
            )

    events.sort(
        key=lambda e: (
            e.video_id,
            e.track_id_a,
            e.track_id_b,
            e.metric_name,
            e.start_frame,
        )
    )
    return events
