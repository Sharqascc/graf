
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.trajectories.conflict_pairs import compute_conflict_pairs
from graf.calibration.homography import project_points
from graf.models.gcn_risk import build_model, has_torch_geometric


def load_tracks(path):
    with open(path) as f:
        tracks = [json.loads(line) for line in f if line.strip()]
    return pd.DataFrame(tracks)

def filter_tracks(df, min_conf=0.4, min_len=5):
    df = df[df["confidence"] >= min_conf].copy()
    lengths = df.groupby("track_id").size()
    valid = lengths[lengths >= min_len].index
    return df[df["track_id"].isin(valid)].copy()

def add_world_coords(df, H, fps=23.98):
    df["x_center"] = (df["bbox_xyxy"].apply(lambda b: b[0]) + df["bbox_xyxy"].apply(lambda b: b[2])) / 2.0
    df["y_bottom"] = df["bbox_xyxy"].apply(lambda b: b[3])
    pts = df[["x_center", "y_bottom"]].to_numpy(dtype=np.float64)
    world = project_points(H, [tuple(p) for p in pts])
    df["x_m"] = world[:, 0]
    df["y_m"] = world[:, 1]
    df["t_sec"] = df["frame_idx"] / fps

    velocities = {}
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        prev_x = prev_y = None
        for idx, row in group.iterrows():
            if prev_x is not None:
                dt = 1.0 / fps
                velocities[row.name] = ((row["x_m"] - prev_x) / dt, (row["y_m"] - prev_y) / dt)
            prev_x, prev_y = row["x_m"], row["y_m"]

    df["vx"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[0])
    df["vy"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[1])
    df["speed_mps"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    return df


def train_fold(train_idx, val_idx, window_ds, labels_tensor, epochs, seed):
    # Set seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    first = window_ds[0]
    model = build_model(in_channels=first.x.size(1), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for idx in train_idx:
            data = window_ds[idx]
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, labels_tensor[idx].unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for idx in val_idx:
            data = window_ds[idx]
            pred = (torch.sigmoid(model(data)) > 0.5).float()
            correct += (pred.item() == labels_tensor[idx].item())
            total += 1
    return correct / total if total else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--graphs_dir", required=True)
    parser.add_argument("--homography_config", required=True)
    parser.add_argument("--output_dir", default="outputs/models_conflict_pairs")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--distance_threshold", type=float, default=5.0)
    parser.add_argument("--min_interaction_frames", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Homography
    with open(args.homography_config) as f:
        H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)

    # Load and prepare tracks
    df = filter_tracks(load_tracks(args.tracks))
    df = add_world_coords(df, H)

    # Compute robust conflicts
    conflicts = compute_conflict_pairs(
        df,
        frame_col="frame_idx",
        track_col="track_id",
        x_col="x_m",
        y_col="y_m",
        speed_col="speed_mps",
        min_interaction_frames=args.min_interaction_frames,
        distance_threshold=args.distance_threshold,
    )

    conflict_frames = set()
    for cp in conflicts:
        conflict_frames.update(cp.frame_indices)

    # Build windows and labels
    window_ds = SpatioTemporalWindowDataset(
        graph_dir=args.graphs_dir,
        window_size=args.window_size,
        stride=args.stride,
    )

    labels = []
    for i in range(len(window_ds)):
        frames = window_ds[i].frame_ids.tolist()
        labels.append(1 if any(fid in conflict_frames for fid in frames) else 0)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # K-fold cross-validation
    np.random.seed(args.seed)
    indices = np.random.permutation(len(window_ds))
    fold_size = len(indices) // args.num_folds
    fold_accs = []

    for fold in range(args.num_folds):
        val_idx = indices[fold * fold_size : (fold + 1) * fold_size]
        train_idx = np.setdiff1d(indices, val_idx)
        acc = train_fold(train_idx, val_idx, window_ds, labels_tensor, args.epochs, args.seed + fold)
        fold_accs.append(acc)
        print(f"Fold {fold+1}/{args.num_folds} | Val Acc: {acc:.3f}")

    mean_acc = float(np.mean(fold_accs))
    std_acc = float(np.std(fold_accs))
    print(f"\nCross-validation accuracy: {mean_acc:.3f} ± {std_acc:.3f}")

    # Save metrics
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "num_windows": len(labels),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "fold_accuracies": [float(x) for x in fold_accs],
        "mean_val_accuracy": mean_acc,
        "std_val_accuracy": std_acc,
        "num_folds": args.num_folds,
        "epochs": args.epochs,
        "seed": args.seed,
        "conflict_pairs_count": len(conflicts),
        "distance_threshold": args.distance_threshold,
        "min_interaction_frames": args.min_interaction_frames,
    }
    with open(out_dir / "metrics_cv.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved cross-validation metrics to {out_dir / 'metrics_cv.json'}")

if __name__ == "__main__":
    main()
