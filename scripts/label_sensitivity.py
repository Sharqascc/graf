"""Label-sensitivity sweep for the VNTraffic TTC-based window labels.

The declared-setup baseline comparison (docs/paper/baselines_vntraffic.md)
showed that with TTC <= 1.5 s and "any frame in the window" labeling, only
29 of 249 windows are negative. That is a nearly degenerate binary task
and no model's AUC is statistically distinguishable from chance.

This script asks: does any reasonable parameterization of the same TTC
label produce a task with enough negatives for reliable evaluation?

Two stages, run in one pass:

  1. Label-distribution sweep. For every combination of
       --ttc-threshold-seconds (default grid 0.5..2.5)
       --ttc-distance-threshold (default grid 2.0..5.0)
       --window-rule in {any, center, majority, all}
     compute the number of positive and negative windows and the
     majority baseline. This stage is cheap (no models) and answers
     "how balanced can we make the labels?"

  2. Model sweep on balanced configs. For configs whose majority rate
     falls in [--balance-lo, --balance-hi] (default [0.55, 0.80]),
     train logistic regression and random forest 5-fold cross-validated
     and record accuracy + AUC. This stage identifies whether any
     balanced config is also learnable.

Writes two artifacts:
  <output_dir>/sweep.json       full grid, one row per config
  <output_dir>/sweep.md         human-readable summary + best configs

Reuses the TTC formula and metric helpers from scripts.evaluate_vntraffic
so the sweep measures the same task as the baseline comparison.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Make both `graf.*` and `scripts.*` importable when this file is
# run as a subprocess (CI, pipeline_status.py). pytest.ini's
# pythonpath setting does not affect subprocesses.
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import pandas as pd
import yaml

from graf.models.baselines import (
    GraphFeatureExtractor,
    LogisticRegressionBaseline,
    RandomForestBaseline,
)
from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from scripts.evaluate_vntraffic import (
    _auc,
    blocked_folds,
)

WINDOW_RULES = ("any", "center", "majority", "all")


def _compute_frame_ttc(
    df: pd.DataFrame,
    *,
    distance_threshold: float,
    closing_rate_threshold: float = 0.5,
) -> dict[int, float]:
    """Per-frame minimum TTC among actor pairs within distance_threshold.

    Returns dict frame_idx -> min TTC (finite). Frames with no qualifying
    pair are absent. This is computed once per distance_threshold and
    reused across all ttc_threshold values, which is what makes the
    sweep cheap.
    """
    out: dict[int, float] = {}
    for frame_idx, frame_df in df.groupby("frame_idx"):
        records = frame_df.to_dict("records")
        frame_min = float("inf")
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
                closing_rate = -float(np.dot(rel_pos, rel_vel))
                rss = float(np.dot(rel_vel, rel_vel))
                if rss <= 1e-9 or closing_rate <= closing_rate_threshold:
                    continue
                ttc = closing_rate / rss
                if np.isfinite(ttc) and 0 < ttc < frame_min:
                    frame_min = ttc
        if frame_min < float("inf"):
            out[int(frame_idx)] = float(frame_min)
    return out


def _apply_window_rule(
    window_ds,
    positive_frames: set[int],
    rule: str,
) -> list[int]:
    """Label every window according to the rule."""
    labels: list[int] = []
    n_windows = len(window_ds)
    for i in range(n_windows):
        frames = [int(f) for f in window_ds[i].frame_ids.tolist()]
        hits = [f in positive_frames for f in frames]
        if rule == "any":
            lab = int(any(hits))
        elif rule == "center":
            lab = int(hits[len(hits) // 2])
        elif rule == "majority":
            lab = int(sum(hits) * 2 > len(hits))
        elif rule == "all":
            lab = int(all(hits))
        else:
            raise ValueError(f"unknown window rule: {rule!r}")
        labels.append(lab)
    return labels


def _evaluate_tabular(
    name: str,
    window_ds,
    labels: list[int],
    folds,
    *,
    seed: int,
) -> dict[str, Any]:
    """Run 5-fold CV with a sklearn baseline; return accuracy + AUC."""
    labels_arr = np.asarray(labels, dtype=np.int64)
    features = np.asarray(
        [
            GraphFeatureExtractor.transform(window_ds[i]).reshape(-1)
            for i in range(len(window_ds))
        ],
        dtype=np.float32,
    )

    fold_acc, fold_auc = [], []
    pooled_scores: list[float] = []
    pooled_labels: list[int] = []
    for k, val_idx in enumerate(folds):
        train_idx = np.concatenate([folds[j] for j in range(len(folds)) if j != k])
        X_train = features[train_idx]
        y_train = labels_arr[train_idx]
        X_val = features[val_idx]
        y_val = labels_arr[val_idx]

        if len(np.unique(y_train)) < 2:
            # All-one-class training fold; can't fit logreg. Skip.
            fold_acc.append(float("nan"))
            fold_auc.append(float("nan"))
            continue

        if name == "logreg":
            model = LogisticRegressionBaseline(random_state=seed + k)
        else:
            model = RandomForestBaseline(random_state=seed + k)
        model.fit(X_train, y_train)

        try:
            proba = model.predict_proba(X_val)
            scores = (
                proba[:, 1]
                if proba.ndim == 2 and proba.shape[1] == 2
                else proba.reshape(-1)
            )
        except Exception:
            scores = np.asarray(model.predict(X_val), dtype=np.float64)
        scores = np.asarray(scores, dtype=np.float64)

        preds = (scores > 0.5).astype(int)
        fold_acc.append(float((preds == y_val).mean()))
        fold_auc.append(_auc(scores, y_val))
        pooled_scores.extend(scores.tolist())
        pooled_labels.extend(y_val.tolist())

    valid_acc = [a for a in fold_acc if a == a]
    valid_auc = [a for a in fold_auc if a == a]
    return {
        "model": name,
        "mean_accuracy": float(np.mean(valid_acc)) if valid_acc else float("nan"),
        "std_accuracy": float(np.std(valid_acc)) if valid_acc else float("nan"),
        "mean_auc": float(np.mean(valid_auc)) if valid_auc else float("nan"),
        "std_auc": float(np.std(valid_auc)) if valid_auc else float("nan"),
        "n_valid_folds": len(valid_acc),
    }


def _write_md(
    path: Path,
    grid: list[dict],
    balanced: list[dict],
) -> None:
    lines: list[str] = []
    lines.append("# Label sensitivity sweep — VNTraffic")
    lines.append("")
    lines.append("Same 10 tracks, same 501 graphs, same blocked 5-fold split.")
    lines.append("The only thing that changes across rows is the label definition.")
    lines.append("")
    lines.append("## Stage 1 — label distribution across the grid")
    lines.append("")
    lines.append("Sorted by majority rate, descending.")
    lines.append("")
    lines.append("| ttc (s) | dist (m) | rule | positives | negatives | majority |")
    lines.append("|---|---|---|---|---|---|")
    for row in sorted(grid, key=lambda r: -r["majority"]):
        lines.append(
            f"| {row['ttc_threshold_seconds']:.1f} "
            f"| {row['distance_threshold']:.1f} "
            f"| {row['window_rule']} "
            f"| {row['num_positive']} "
            f"| {row['num_negative']} "
            f"| {row['majority']:.3f} |"
        )

    lines.append("")
    lines.append("## Stage 2 — model sweep on balanced configs")
    lines.append("")
    lines.append(
        "Configs whose majority falls in "
        f"[{grid[0].get('_balance_lo', 0.55)}, "
        f"{grid[0].get('_balance_hi', 0.80)}] are not shown here; see JSON "
        "for the full grid. This stage reports every config in that range "
        "without reordering, filter, or cherry-pick."
    )
    lines.append("")
    if not balanced:
        lines.append("_No configs fell in the balanced range._")
    else:
        lines.append(
            "| ttc (s) | dist (m) | rule | pos | neg | majority "
            "| logreg acc | logreg auc | rf acc | rf auc |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for row in balanced:
            lr = next((m for m in row["models"] if m["model"] == "logreg"), None)
            rf = next((m for m in row["models"] if m["model"] == "rf"), None)
            cells = [
                f"{row['ttc_threshold_seconds']:.1f}",
                f"{row['distance_threshold']:.1f}",
                row["window_rule"],
                str(row["num_positive"]),
                str(row["num_negative"]),
                f"{row['majority']:.3f}",
            ]
            if lr:
                cells.append(f"{lr['mean_accuracy']:.3f}")
                cells.append(f"{lr['mean_auc']:.3f}")
            else:
                cells.extend(["-", "-"])
            if rf:
                cells.append(f"{rf['mean_accuracy']:.3f}")
                cells.append(f"{rf['mean_auc']:.3f}")
            else:
                cells.extend(["-", "-"])
            lines.append("| " + " | ".join(cells) + " |")

    path.write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracks", required=True)
    p.add_argument("--graphs_dir", required=True)
    p.add_argument("--homography_config", required=True)
    p.add_argument("--output_dir", default="outputs/label_sensitivity")
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--min_interaction_frames", type=int, default=3)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--ttc-grid",
        default="0.5,1.0,1.5,2.0,2.5",
        help="Comma-separated TTC thresholds in seconds.",
    )
    p.add_argument(
        "--distance-grid",
        default="2.0,3.0,4.0,5.0",
        help="Comma-separated distance thresholds in metres.",
    )
    p.add_argument(
        "--balance-lo",
        type=float,
        default=0.55,
        help="Minimum majority rate to run models on (Stage 2).",
    )
    p.add_argument(
        "--balance-hi",
        type=float,
        default=0.80,
        help="Maximum majority rate to run models on (Stage 2).",
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    t0 = time.time()

    ttc_grid = [float(x) for x in args.ttc_grid.split(",") if x.strip()]
    dist_grid = [float(x) for x in args.distance_grid.split(",") if x.strip()]

    # Load data once
    with open(args.homography_config) as f:
        H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)
    df = filter_tracks(load_tracks(args.tracks))
    df = add_world_coords(df, H, fps=args.fps)
    print(f"tracks: {df['track_id'].nunique()}  frames: {df['frame_idx'].nunique()}")

    # Build windows once (label-independent)
    from graf.data.graph_dataset import SpatioTemporalWindowDataset

    window_ds = SpatioTemporalWindowDataset(
        graph_dir=args.graphs_dir,
        window_size=args.window_size,
        stride=args.stride,
    )
    n_windows = len(window_ds)
    print(f"windows: {n_windows}")
    frame_ids_per_window = [window_ds[i].frame_ids.tolist() for i in range(n_windows)]
    folds = blocked_folds(frame_ids_per_window, args.num_folds)

    # Stage 1: sweep the label grid
    print("\nStage 1 — label sweep")
    print("=" * 70)
    grid: list[dict] = []
    per_distance_ttc: dict[float, dict[int, float]] = {}
    for dist in dist_grid:
        per_distance_ttc[dist] = _compute_frame_ttc(df, distance_threshold=dist)
        print(
            f"  computed frame TTC for dist<={dist} m: "
            f"{len(per_distance_ttc[dist])} frames with a TTC event"
        )

    for ttc in ttc_grid:
        for dist in dist_grid:
            frame_ttc = per_distance_ttc[dist]
            positive_frames = {f for f, v in frame_ttc.items() if v <= ttc}
            for rule in WINDOW_RULES:
                labels: list[int] = _apply_window_rule(window_ds, positive_frames, rule)
                n_pos = sum(labels)
                n_neg = n_windows - n_pos
                majority = max(n_pos, n_neg) / n_windows if n_windows else 0.0
                row: dict[str, Any] = {
                    "ttc_threshold_seconds": ttc,
                    "distance_threshold": dist,
                    "window_rule": rule,
                    "num_positive": int(n_pos),
                    "num_negative": int(n_neg),
                    "majority": float(majority),
                    "labels": labels,
                    "models": [],
                }
                grid.append(row)

    # Stage 2: model sweep on balanced configs
    balanced = [r for r in grid if args.balance_lo <= r["majority"] <= args.balance_hi]
    print(
        f"\nStage 2 — {len(balanced)} configs in majority range "
        f"[{args.balance_lo}, {args.balance_hi}]"
    )

    for row in balanced:
        print(
            f"\n  ttc<={row['ttc_threshold_seconds']} "
            f"dist<={row['distance_threshold']} "
            f"rule={row['window_rule']}  "
            f"pos={row['num_positive']} neg={row['num_negative']} "
            f"maj={row['majority']:.3f}"
        )
        for model_name in ("logreg", "rf"):
            m = _evaluate_tabular(
                model_name, window_ds, row["labels"], folds, seed=args.seed
            )
            row["models"].append(m)
            print(
                f"    {model_name:8s}  "
                f"acc={m['mean_accuracy']:.3f}±{m['std_accuracy']:.3f}  "
                f"auc={m['mean_auc']:.3f}±{m['std_auc']:.3f}  "
                f"({m['n_valid_folds']} valid folds)"
            )

    # Strip labels from the JSON (they're big and derivable)
    for row in grid:
        row.pop("labels", None)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload = {
        "setup": {
            "tracks": args.tracks,
            "graphs_dir": args.graphs_dir,
            "num_windows": n_windows,
            "num_folds": args.num_folds,
            "window_size": args.window_size,
            "stride": args.stride,
            "ttc_grid": ttc_grid,
            "distance_grid": dist_grid,
            "window_rules": list(WINDOW_RULES),
            "balance_range": [args.balance_lo, args.balance_hi],
            "seed": args.seed,
        },
        "grid": grid,
        "runtime_seconds": round(time.time() - t0, 1),
    }
    (out / "sweep.json").write_text(json.dumps(payload, indent=2) + "\n")
    _write_md(out / "sweep.md", grid, balanced)
    print(f"\nWrote {out / 'sweep.json'}")
    print(f"Wrote {out / 'sweep.md'}")
    print(f"Runtime: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
