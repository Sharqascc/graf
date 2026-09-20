"""Alternative window-level label strategies for the TTC-based task.

The default label strategy ("any") defines a window as positive if any
frame in the window contains a TTC event below the threshold. At scale
this saturates: with 81 actors in frame, some pair is always critical
and every window is positive (see docs/paper/baselines_vntraffic_all81.md).

This module provides three alternatives that try to keep the label
discriminative when the actor count is large:

    any             default. Window positive if any frame has a TTC
                    event below the threshold.
    pair_fraction   Window positive if the maximum, over frames in the
                    window, of (critical pairs / total pairs) exceeds
                    --fraction-threshold. Normalizes by actor count.
    sustained       Window positive if at least --run-length consecutive
                    frames each have a TTC event below the threshold.
                    Filters single-frame spikes.
    min_ttc         Window positive only if the *minimum* TTC across all
                    frames and pairs in the window is below the
                    threshold. Stricter than "any".

Every strategy is a function with the same signature so the caller can
dispatch on --label-strategy without special cases:

    fn(df, window_ds, *, ttc_threshold_seconds, distance_threshold,
       closing_rate_threshold, **strategy_kwargs) -> list[int]

"actor_level" is deliberately not implemented: it would change the label
shape from per-window to per-(window, actor) and require reworking every
downstream helper. It is left for a future change.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

STRATEGIES = ("any", "pair_fraction", "sustained", "min_ttc")


def _frame_ttc_stats(
    df: pd.DataFrame,
    *,
    ttc_threshold_seconds: float,
    distance_threshold: float,
    closing_rate_threshold: float,
) -> pd.DataFrame:
    """Per-frame pair-level TTC summary.

    Returns a DataFrame indexed by frame_idx with columns:
        n_pairs      number of actor pairs within distance_threshold
        n_critical   pairs whose TTC is in (0, ttc_threshold_seconds]
        min_ttc      the smallest TTC among critical pairs, or inf
    """
    rows: list[dict] = []
    for frame_idx, frame_df in df.groupby("frame_idx"):
        records = frame_df.to_dict("records")
        n_pairs = 0
        n_critical = 0
        min_ttc = float("inf")
        n = len(records)
        for i in range(n):
            a = records[i]
            for j in range(i + 1, n):
                b = records[j]
                rel_pos = np.array([b["x_m"] - a["x_m"], b["y_m"] - a["y_m"]])
                rel_vel = np.array([b["vx"] - a["vx"], b["vy"] - a["vy"]])
                dist = float(np.linalg.norm(rel_pos))
                if dist >= distance_threshold:
                    continue
                n_pairs += 1
                closing_rate = -float(np.dot(rel_pos, rel_vel))
                rss = float(np.dot(rel_vel, rel_vel))
                if rss <= 1e-9 or closing_rate <= closing_rate_threshold:
                    continue
                ttc = closing_rate / rss
                if np.isfinite(ttc) and 0 < ttc <= ttc_threshold_seconds:
                    n_critical += 1
                    if ttc < min_ttc:
                        min_ttc = ttc
        rows.append(
            {
                "frame_idx": int(frame_idx),
                "n_pairs": n_pairs,
                "n_critical": n_critical,
                "min_ttc": min_ttc,
            }
        )
    return pd.DataFrame(rows).set_index("frame_idx")


def _window_frames(window_ds) -> list[list[int]]:
    """Return the frame_ids of every window as a list of int lists."""
    return [
        [int(f) for f in window_ds[i].frame_ids.tolist()] for i in range(len(window_ds))
    ]


# ------------------------------------------------------------------
# Strategies
# ------------------------------------------------------------------


def label_any(
    df: pd.DataFrame,
    window_ds,
    *,
    ttc_threshold_seconds: float = 1.5,
    distance_threshold: float = 3.0,
    closing_rate_threshold: float = 0.5,
    **_: object,
) -> list[int]:
    """Default: window positive if any frame has >=1 critical pair."""
    stats = _frame_ttc_stats(
        df,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
    )
    critical_frames = set(stats[stats["n_critical"] > 0].index.tolist())
    return [
        1 if any(int(f) in critical_frames for f in w) else 0
        for w in _window_frames(window_ds)
    ]


def label_pair_fraction(
    df: pd.DataFrame,
    window_ds,
    *,
    ttc_threshold_seconds: float = 1.5,
    distance_threshold: float = 3.0,
    closing_rate_threshold: float = 0.5,
    fraction_threshold: float = 0.2,
    **_: object,
) -> list[int]:
    """Window positive if the max per-frame critical-pair fraction
    across the window is >= fraction_threshold.

    Normalizes by the number of nearby pairs in the frame, so
    crowded scenes do not automatically produce positives.
    """
    stats = _frame_ttc_stats(
        df,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
    )
    # Fraction is 0 whenever n_pairs==0 (avoid divide-by-zero)
    frac = np.where(
        stats["n_pairs"] > 0,
        stats["n_critical"] / stats["n_pairs"].clip(lower=1),
        0.0,
    )
    frame_fraction = dict(zip(stats.index.tolist(), frac.tolist(), strict=True))

    labels: list[int] = []
    for w in _window_frames(window_ds):
        max_frac = max((frame_fraction.get(int(f), 0.0) for f in w), default=0.0)
        labels.append(1 if max_frac >= fraction_threshold else 0)
    return labels


def label_sustained(
    df: pd.DataFrame,
    window_ds,
    *,
    ttc_threshold_seconds: float = 1.5,
    distance_threshold: float = 3.0,
    closing_rate_threshold: float = 0.5,
    run_length: int = 3,
    **_: object,
) -> list[int]:
    """Window positive if at least run_length consecutive frames in the
    window each contain a critical pair.

    Filters single-frame spikes that dominate at high actor counts.
    """
    if run_length < 1:
        raise ValueError(f"run_length must be >= 1, got {run_length}")
    stats = _frame_ttc_stats(
        df,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
    )
    critical_frames = set(stats[stats["n_critical"] > 0].index.tolist())

    labels: list[int] = []
    for w in _window_frames(window_ds):
        # Longest run of consecutive-in-window frames that are all critical
        best = 0
        cur = 0
        for f in w:
            if int(f) in critical_frames:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        labels.append(1 if best >= run_length else 0)
    return labels


def label_min_ttc(
    df: pd.DataFrame,
    window_ds,
    *,
    ttc_threshold_seconds: float = 1.5,
    distance_threshold: float = 3.0,
    closing_rate_threshold: float = 0.5,
    **_: object,
) -> list[int]:
    """Window positive only if the minimum TTC across all frames and
    pairs in the window is <= ttc_threshold_seconds.

    Equivalent to "any" at the frame level (a single critical pair is
    enough), but the threshold now applies to the extreme TTC rather
    than to a per-frame flag. With a lower ttc_threshold this is much
    stricter than "any".
    """
    stats = _frame_ttc_stats(
        df,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
    )
    frame_min = dict(zip(stats.index.tolist(), stats["min_ttc"].tolist(), strict=True))

    labels: list[int] = []
    for w in _window_frames(window_ds):
        w_min = min(
            (frame_min.get(int(f), float("inf")) for f in w), default=float("inf")
        )
        labels.append(1 if w_min <= ttc_threshold_seconds else 0)
    return labels


_DISPATCH: dict[str, Callable[..., list[int]]] = {
    "any": label_any,
    "pair_fraction": label_pair_fraction,
    "sustained": label_sustained,
    "min_ttc": label_min_ttc,
}


def apply_strategy(
    strategy: str,
    df: pd.DataFrame,
    window_ds,
    *,
    ttc_threshold_seconds: float = 1.5,
    distance_threshold: float = 3.0,
    closing_rate_threshold: float = 0.5,
    **strategy_kwargs: object,
) -> list[int]:
    """Dispatch to the named strategy, raising on unknown names."""
    fn = _DISPATCH.get(strategy)
    if fn is None:
        raise ValueError(
            f"unknown label strategy {strategy!r}; choose from {STRATEGIES}"
        )
    return fn(
        df,
        window_ds,
        ttc_threshold_seconds=ttc_threshold_seconds,
        distance_threshold=distance_threshold,
        closing_rate_threshold=closing_rate_threshold,
        **strategy_kwargs,
    )
