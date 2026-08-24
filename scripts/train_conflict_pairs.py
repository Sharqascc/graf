
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
from graf.trajectories.conflict_pairs import compute_conflict_pairs, find_nearby_pairs
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

def add_world_coords(df, H):
    df["x_center"] = (df["bbox_xyxy"].apply(lambda b: b[0]) + df["bbox_xyxy"].apply(lambda b: b[2])) / 2.0
    df["y_bottom"] = df["bbox_xyxy"].apply(lambda b: b[3])
    pts = df[["x_center", "y_bottom"]].to_numpy(dtype=np.float64)
    world = project_points(H, [tuple(p) for p in pts])
    df["x_m"] = world[:, 0]
    df["y_m"] = world[:, 1]
    df["t_sec"] = df["frame_idx"] / 23.98

    velocities = {}
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        prev_x = prev_y = None
        for idx, row in group.iterrows():
            if prev_x is not None:
                dt = 1.0 / 23.98
                velocities[row.name] = ((row["x_m"] - prev_x) / dt, (row["y_m"] - prev_y) / dt)
            prev_x, prev_y = row["x_m"], row["y_m"]

    df["vx"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[0])
    df["vy"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[1])
    df["speed_mps"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    return df


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
    args = parser.parse_args()

    # Load homography
    with open(args.homography_config) as f:
        H = np.array(yaml.safe_load(f)["H"], dtype=np.float64)

    # Load and prepare tracks
    df = load_tracks(args.tracks)
    df = filter_tracks(df)
    df = add_world_coords(df, H)

    # Compute robust conflict pairs
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
    indices = list(range(len(window_ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx, val_idx = indices[:split], indices[split:]

    if not has_torch_geometric or len(window_ds) == 0:
        print("Torch Geometric not available or no windows.")
        return

    first = window_ds[0]
    model = build_model(in_channels=first.x.size(1), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
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
        correct = total = 0
        with torch.no_grad():
            for idx in val_idx:
                data = window_ds[idx]
                pred = (torch.sigmoid(model(data)) > 0.5).float()
                correct += (pred.item() == labels_tensor[idx].item())
                total += 1
        acc = correct / total if total else 0.0
        best_acc = max(best_acc, acc)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_idx):.4f} | Val Acc: {acc:.3f}")

    print(f"\nBest validation accuracy: {best_acc:.3f}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "gcn_conflict_pairs.pt")
    metrics = {
        "num_windows": len(labels),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "best_val_accuracy": float(best_acc),
        "conflict_pairs_count": len(conflicts),
        "epochs": args.epochs,
        "distance_threshold": args.distance_threshold,
        "min_interaction_frames": args.min_interaction_frames,
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved model and metrics to {out_dir}")

if __name__ == "__main__":
    main()
