"""Conflict-pair supervised training for graph-level risk classification."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from ..calibration.homography import project_points
from ..data.graph_dataset import SpatioTemporalWindowDataset
from ..models.gcn_risk import build_model
from ..trajectories.conflict_pairs import compute_conflict_pairs


def load_tracks(path: str | Path) -> pd.DataFrame:
    """Load a JSONL tracks file into a DataFrame."""
    with open(path) as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(tracks)


def filter_tracks(
    df: pd.DataFrame, min_conf: float = 0.4, min_len: int = 5
) -> pd.DataFrame:
    """Drop low-confidence detections and short tracks."""
    df = df[df["confidence"] >= min_conf].copy()
    lengths = df.groupby("track_id").size()
    valid = lengths[lengths >= min_len].index
    return df[df["track_id"].isin(valid)].copy()


def add_world_coords(
    df: pd.DataFrame, H: np.ndarray, fps: float = 23.98
) -> pd.DataFrame:
    """Project pixel bboxes into world coordinates and compute velocities."""
    df = df.copy()
    df["x_center"] = (
        df["bbox_xyxy"].apply(lambda b: b[0])
        + df["bbox_xyxy"].apply(lambda b: b[2])
    ) / 2.0
    df["y_bottom"] = df["bbox_xyxy"].apply(lambda b: b[3])
    pts = df[["x_center", "y_bottom"]].to_numpy(dtype=np.float64)
    world = project_points(H, [tuple(p) for p in pts])
    df["x_m"] = world[:, 0]
    df["y_m"] = world[:, 1]
    df["t_sec"] = df["frame_idx"] / fps

    velocities: dict = {}
    for _, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        prev_x = prev_y = None
        for idx, row in group.iterrows():
            if prev_x is not None:
                dt = 1.0 / fps
                velocities[idx] = (
                    (row["x_m"] - prev_x) / dt,
                    (row["y_m"] - prev_y) / dt,
                )
            prev_x, prev_y = row["x_m"], row["y_m"]

    df["vx"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[0])
    df["vy"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[1])
    df["speed_mps"] = np.sqrt(df["vx"] ** 2 + df["vy"] ** 2)
    return df


def train_fold(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    window_ds,
    labels: torch.Tensor,
    epochs: int,
    seed: int,
) -> float:
    """Train one fold and return validation accuracy."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    first = window_ds[0]
    model = build_model(in_channels=first.x.size(1), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    for _ in range(epochs):
        model.train()
        for idx in train_idx:
            data = window_ds[idx]
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels[idx].unsqueeze(0))
            loss.backward()
            optimizer.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for idx in val_idx:
            data = window_ds[idx]
            pred = (torch.sigmoid(model(data)) > 0.5).float()
            correct += int(pred.item() == labels[idx].item())
            total += 1
    return correct / total if total else 0.0


def run_cross_validation(
    *,
    tracks_path: str | Path,
    graphs_dir: str | Path,
    homography_config: str | Path,
    output_dir: str | Path = "outputs/models_conflict_pairs",
    window_size: int = 5,
    stride: int = 2,
    distance_threshold: float = 5.0,
    min_interaction_frames: int = 3,
    epochs: int = 50,
    num_folds: int = 5,
    seed: int = 42,
    fps: float = 23.98,
) -> dict:
    """Run k-fold CV over temporal windows labelled by conflict presence.

    Returns the metrics dict that is also written to ``metrics_cv.json``.
    """
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
        1 if any(fid in conflict_frames for fid in window_ds[i].frame_ids.tolist()) else 0
        for i in range(len(window_ds))
    ]
    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    np.random.seed(seed)
    indices = np.random.permutation(len(window_ds))
    fold_size = len(indices) // num_folds
    fold_accs = []

    for fold in range(num_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        acc = train_fold(
            train_idx, val_idx, window_ds, labels_tensor, epochs, seed + fold
        )
        fold_accs.append(acc)
        print(f"Fold {fold+1}/{num_folds} | Val Acc: {acc:.3f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"\nCross-validation accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_windows": len(labels),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "fold_accuracies": [float(x) for x in fold_accs],
        "mean_val_accuracy": mean_acc,
        "std_val_accuracy": std_acc,
        "num_folds": num_folds,
        "epochs": epochs,
        "seed": seed,
        "conflict_pairs_count": len(conflicts),
        "distance_threshold": distance_threshold,
        "min_interaction_frames": min_interaction_frames,
    }
    with open(out_dir / "metrics_cv.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {out_dir / 'metrics_cv.json'}")
    return metrics
