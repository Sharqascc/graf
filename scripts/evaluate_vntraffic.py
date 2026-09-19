"""Reproducible k-fold evaluation on VNTraffic-derived graphs.

Recreates the setup in docs/real_data_evaluation.md and
docs/real_data_metrics.json: windowed SSM labels, GCN risk model,
k-fold CV with configurable batch size and class weighting.

Differences from graf.training.conflict_pairs.run_cross_validation:

  * --split blocked (default) or random. Blocked assigns contiguous
    frame-ordered blocks of windows to folds, preventing the
    overlapping-window leak that a random split allows. Random matches
    the original ad-hoc run whose output is in
    docs/real_data_metrics.json.
  * --pos-weight and --batch-size so that setup (pos_weight=2.0,
    batch_size=16) can be reproduced.
  * Reports per-fold F1, AUC, majority-class rate, window-overlap
    leakage, pooled accuracy/AUC, and a binomial significance test
    versus majority in a single pass.

Requires graphs and tracks prepared by scripts/prepare_vntraffic.py.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
import yaml

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.models.gcn_risk import build_model
from graf.training.conflict_pairs import (
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from graf.trajectories.conflict_pairs import compute_conflict_pairs


def _f1(preds: np.ndarray, labels: np.ndarray) -> float:
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Average-rank AUC with tie handling (Hanley-McNeil)."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg])
    order = np.argsort(combined, kind="mergesort")
    sorted_vals = combined[order]
    n = len(sorted_vals)
    ranks = np.empty(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def _leakage_count(train_idx, val_idx, frame_ids_per_window):
    train_frames = set()
    for i in train_idx:
        train_frames.update(frame_ids_per_window[int(i)])
    leaked = sum(
        1
        for i in val_idx
        if any(f in train_frames for f in frame_ids_per_window[int(i)])
    )
    return int(leaked), int(len(val_idx))


def _binomial_vs_majority(correct: int, total: int, majority_rate: float) -> float:
    if total <= 0:
        return 1.0
    if total <= 500:
        p = 0.0
        for k in range(correct, total + 1):
            p += (
                comb(total, k)
                * (majority_rate**k)
                * ((1 - majority_rate) ** (total - k))
            )
        return float(min(1.0, p))
    mu = total * majority_rate
    sd = math.sqrt(total * majority_rate * (1 - majority_rate))
    if sd == 0:
        return 1.0
    z = (correct - mu) / sd
    return float(0.5 * math.erfc(z / math.sqrt(2.0)))


def build_labels(
    tracks_path,
    graphs_dir,
    homography_config,
    window_size,
    stride,
    distance_threshold,
    min_interaction_frames,
    fps,
):
    with open(homography_config) as f:
        H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)
    df = filter_tracks(load_tracks(tracks_path))
    df = add_world_coords(df, H, fps=fps)
    conflicts = compute_conflict_pairs(
        df,
        frame_col="frame_idx",
        track_col="track_id",
        x_col="x_m",
        y_col="y_m",
        speed_col="speed_mps",
        min_interaction_frames=min_interaction_frames,
        distance_threshold=distance_threshold,
    )
    conflict_frames: set = set()
    for cp in conflicts:
        conflict_frames.update(cp.frame_indices)
    window_ds = SpatioTemporalWindowDataset(
        graph_dir=graphs_dir, window_size=window_size, stride=stride
    )
    labels = [
        1
        if any(fid in conflict_frames for fid in window_ds[i].frame_ids.tolist())
        else 0
        for i in range(len(window_ds))
    ]
    return window_ds, labels


def blocked_folds(frame_ids_per_window, num_folds: int):
    order = sorted(
        range(len(frame_ids_per_window)),
        key=lambda i: min(frame_ids_per_window[i]),
    )
    block = max(1, len(order) // num_folds)
    folds = []
    for k in range(num_folds):
        lo = k * block
        hi = (k + 1) * block if k < num_folds - 1 else len(order)
        folds.append(np.array(order[lo:hi], dtype=int))
    return folds


def random_folds(n_items: int, num_folds: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_items)
    return [np.array(f, dtype=int) for f in np.array_split(idx, num_folds)]


def train_one_fold(
    window_ds,
    labels,
    train_idx,
    val_idx,
    epochs,
    batch_size,
    lr,
    pos_weight,
    seed,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    first = window_ds[0]
    model = build_model(in_channels=first.x.size(1), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if pos_weight is None:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        pw = torch.tensor([float(pos_weight)], dtype=torch.float32)
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pw)

    train_labels = torch.tensor(
        [labels[int(i)] for i in train_idx], dtype=torch.float32
    )

    for _ in range(epochs):
        model.train()
        perm = np.random.permutation(len(train_idx))
        for start in range(0, len(perm), batch_size):
            batch = perm[start : start + batch_size]
            optimizer.zero_grad()
            losses = []
            for b in batch:
                data = window_ds[int(train_idx[int(b)])]
                out = model(data)
                losses.append(criterion(out, train_labels[int(b)].unsqueeze(0)))
            loss = torch.stack(losses).mean()
            loss.backward()
            optimizer.step()

    model.eval()
    val_scores = []
    val_labels = []
    with torch.no_grad():
        for idx in val_idx:
            data = window_ds[int(idx)]
            score = float(torch.sigmoid(model(data)).item())
            val_scores.append(score)
            val_labels.append(int(labels[int(idx)]))
    return np.array(val_scores), np.array(val_labels)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--tracks", required=True)
    p.add_argument("--graphs_dir", required=True)
    p.add_argument("--homography_config", required=True)
    p.add_argument("--output_dir", default="outputs/models_vntraffic")
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--distance_threshold", type=float, default=5.0)
    p.add_argument("--min_interaction_frames", type=int, default=3)
    p.add_argument("--fps", type=float, default=23.98)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--pos_weight", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split", choices=["blocked", "random"], default="blocked")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    window_ds, labels = build_labels(
        args.tracks,
        args.graphs_dir,
        args.homography_config,
        args.window_size,
        args.stride,
        args.distance_threshold,
        args.min_interaction_frames,
        args.fps,
    )
    n = len(window_ds)
    if n == 0:
        raise SystemExit("No windows found — check the graphs_dir.")
    n_pos = int(sum(labels))
    majority = max(n_pos, n - n_pos) / n
    print(f"Windows: {n}  positives: {n_pos}  majority: {majority:.3f}")

    frame_ids_per_window = [window_ds[i].frame_ids.tolist() for i in range(n)]

    if args.split == "blocked":
        folds = blocked_folds(frame_ids_per_window, args.num_folds)
    else:
        folds = random_folds(n, args.num_folds, args.seed)

    fold_acc, fold_f1, fold_auc, fold_maj = [], [], [], []
    leaks = []
    pooled_scores = []
    pooled_labels = []

    for k in range(args.num_folds):
        val_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(args.num_folds) if j != k])
        n_leaked, n_val = _leakage_count(train_idx, val_idx, frame_ids_per_window)
        leaks.append(
            {"fold": k, "val_windows_shared_frames": n_leaked, "val_windows": n_val}
        )
        scores, val_labels = train_one_fold(
            window_ds,
            labels,
            train_idx,
            val_idx,
            args.epochs,
            args.batch_size,
            args.lr,
            args.pos_weight,
            args.seed + k,
        )
        preds = (scores > 0.5).astype(int)
        acc = float((preds == val_labels).mean()) if len(val_labels) else 0.0
        f1 = _f1(preds, val_labels)
        auc = _auc(scores, val_labels)
        maj = max(val_labels.mean(), 1 - val_labels.mean()) if len(val_labels) else 0.0
        fold_acc.append(acc)
        fold_f1.append(f1)
        fold_auc.append(auc)
        fold_maj.append(float(maj))
        pooled_scores.extend(scores.tolist())
        pooled_labels.extend(val_labels.tolist())
        print(
            f"Fold {k + 1}/{args.num_folds}: acc={acc:.3f}  f1={f1:.3f}  auc={auc:.3f}  maj={maj:.3f}"
        )

    pooled_correct = int(
        sum(
            1
            for s, lbl in zip(pooled_scores, pooled_labels)
            if int(s > 0.5) == int(lbl)
        )
    )
    pooled_total = len(pooled_labels)
    pooled_acc = pooled_correct / max(pooled_total, 1)
    p_value = _binomial_vs_majority(pooled_correct, pooled_total, majority)

    summary = {
        "num_windows": n,
        "num_positive": n_pos,
        "num_negative": n - n_pos,
        "majority_baseline": float(majority),
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "pos_weight": args.pos_weight,
        "split": args.split,
        "seed": args.seed,
        "fold_acc": fold_acc,
        "fold_f1": fold_f1,
        "fold_auc": fold_auc,
        "fold_majority": fold_maj,
        "mean_accuracy": float(np.mean(fold_acc)),
        "std_accuracy": float(np.std(fold_acc)),
        "mean_f1": float(np.mean(fold_f1)),
        "mean_auc": float(np.nanmean(fold_auc)),
        "leakage": leaks,
        "diagnostics": {
            "pooled_accuracy": float(pooled_acc),
            "pooled_auc": float(_auc(np.array(pooled_scores), np.array(pooled_labels))),
            "p_value_vs_majority": float(p_value),
            "beats_majority_p05": bool(p_value < 0.05),
        },
    }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.json"
    metrics_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
