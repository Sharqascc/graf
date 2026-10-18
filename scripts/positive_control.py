"""Positive control for the label pipeline (item 2 of PLAN.md).

Checks five things that must all hold for the paper to be
interpretable:

  1. Is data.y populated on per-frame graphs?
  2. Does data.y propagate into window graphs via temporal.py?
  3. Does the harness label_sustained match an independent
     reimplementation from the same _frame_ttc_stats?
  4. Does a decision tree on n_critical_frames recover the
     harness label at ~1.0 accuracy?
  5. Does a tree on min_ttc also recover? (Different rule.)

Writes positive_control.json to --output_dir.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import yaml
from sklearn.tree import DecisionTreeClassifier

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from scripts.label_strategies import (
    _frame_ttc_stats,
    apply_strategy,
)


def _window_frames(window_ds) -> list[list[int]]:
    out: list[list[int]] = []
    for w in window_ds.windows:
        frames = [
            int(getattr(window_ds.base_dataset.get(i), "frame_id", i))
            for i in w
        ]
        out.append(frames)
    return out


def _check_data_y(window_ds, n_sample: int = 50) -> dict:
    total = len(window_ds)
    n = min(n_sample, total)
    populated = 0
    values: list[float] = []
    for i in range(n):
        w = window_ds[i]
        y = getattr(w, "y", None)
        if y is None:
            continue
        try:
            arr = np.asarray(y).flatten()
        except Exception:
            continue
        if arr.size == 0:
            continue
        populated += 1
        values.append(float(arr[0]))
    return {
        "n_windows_total": total,
        "n_sampled": n,
        "n_with_y": populated,
        "y_values_distinct": sorted({round(v, 3) for v in values}),
    }


def _check_frame_y(graphs_dir: Path, n_sample: int = 50) -> dict:
    import torch
    files = sorted(graphs_dir.glob("*.pt"))[:n_sample]
    populated = 0
    values: list[float] = []
    for f in files:
        g = torch.load(f, weights_only=False)
        y = getattr(g, "y", None)
        if y is None:
            continue
        arr = np.asarray(y).flatten()
        if arr.size == 0:
            continue
        populated += 1
        values.append(float(arr[0]))
    return {
        "n_frames_sampled": len(files),
        "n_with_y": populated,
        "y_values_distinct": sorted({round(v, 3) for v in values}),
    }


def _sustained_labels(
    stats,
    windows: list[list[int]],
    run_length: int,
) -> list[int]:
    critical = set(stats[stats["n_critical"] > 0].index.tolist())
    labels: list[int] = []
    for w in windows:
        best = 0
        cur = 0
        for f in w:
            if int(f) in critical:
                cur += 1
                best = max(best, cur)
            else:
                cur = 0
        labels.append(1 if best >= run_length else 0)
    return labels


def _tree_accuracy(X: np.ndarray, y: np.ndarray) -> dict:
    if X.size == 0 or len(set(y.tolist())) < 2:
        return {"accuracy": None, "note": "degenerate labels"}
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X, y)
    acc = float(clf.score(X, y))
    threshold = None
    try:
        threshold = float(clf.tree_.threshold[0])
    except Exception:
        pass
    return {"accuracy": acc, "threshold": threshold}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tracks", required=True)
    ap.add_argument("--graphs_dir", required=True)
    ap.add_argument("--homography_config", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--window_size", type=int, default=5)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--run_length", type=int, default=5)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--ttc_threshold", type=float, default=1.5)
    ap.add_argument("--distance_threshold", type=float, default=3.0)
    args = ap.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {"setup": vars(args)}

    graphs_path = Path(args.graphs_dir)
    print(f"Checking per-frame y in {graphs_path} ...")
    report["frame_y"] = _check_frame_y(graphs_path)
    print(f"  {report['frame_y']}")

    print("Loading windows ...")
    window_ds = SpatioTemporalWindowDataset(
        graphs_path,
        window_size=args.window_size,
        stride=args.stride,
    )
    print(f"  {len(window_ds)} windows")

    print("Checking window y ...")
    report["window_y"] = _check_data_y(window_ds)
    print(f"  {report['window_y']}")

    print("Loading tracks ...")
    df = filter_tracks(load_tracks(args.tracks))
    with Path(args.homography_config).open() as f:
        H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)
    df = add_world_coords(df, H, fps=args.fps)

    print("Computing TTC stats ...")
    stats = _frame_ttc_stats(
        df,
        ttc_threshold_seconds=args.ttc_threshold,
        distance_threshold=args.distance_threshold,
        closing_rate_threshold=0.5,
    )

    windows = _window_frames(window_ds)
    print(f"  {len(windows)} windows")

    harness = apply_strategy(
        "sustained",
        df,
        window_ds,
        ttc_threshold_seconds=args.ttc_threshold,
        distance_threshold=args.distance_threshold,
        run_length=args.run_length,
    )
    independent = _sustained_labels(stats, windows, args.run_length)
    agree = sum(int(a) == int(b) for a, b in zip(harness, independent))
    report["sustained_agreement"] = {
        "n": len(harness),
        "n_match": agree,
        "fraction": agree / max(1, len(harness)),
    }
    print(f"  sustained agreement: {report['sustained_agreement']}")

    critical = set(stats[stats["n_critical"] > 0].index.tolist())
    n_crit = np.array(
        [sum(1 for f in w if int(f) in critical) for w in windows],
        dtype=float,
    )
    y_true = np.array(harness, dtype=int)
    report["tree_on_n_critical"] = _tree_accuracy(n_crit, y_true)
    print(f"  tree on n_critical: {report['tree_on_n_critical']}")

    frame_min: dict[int, float] = {}
    for idx, row in stats.iterrows():
        frame_min[int(idx)] = float(row["min_ttc"])
    window_min: list[float] = []
    for w in windows:
        vals = [frame_min.get(int(f), float("inf")) for f in w]
        window_min.append(min(vals) if vals else float("inf"))
    finite = [v for v in window_min if np.isfinite(v)]
    big = (max(finite) + 1.0) if finite else 1e9
    x_min = np.array([v if np.isfinite(v) else big for v in window_min], dtype=float)
    report["tree_on_min_ttc"] = _tree_accuracy(x_min.reshape(-1, 1), y_true)
    print(f"  tree on min_ttc: {report['tree_on_min_ttc']}")

    (out / "positive_control.json").write_text(
        json.dumps(report, indent=2) + chr(10)
    )
    print("wrote positive_control.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
