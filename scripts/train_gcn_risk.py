
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from graf.data.graph_dataset import SpatioTemporalWindowDataset
from graf.models.gcn_risk import build_model, has_torch_geometric
from graf.utils.io import ensure_dir, write_json


def load_tracks(path: str) -> pd.DataFrame:
    tracks = []
    with open(path) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))
    df = pd.DataFrame(tracks)
    return df


def filter_tracks(df: pd.DataFrame, min_conf: float = 0.4, min_len: int = 5) -> pd.DataFrame:
    df = df[df["confidence"] >= min_conf].copy()
    lengths = df.groupby("track_id").size()
    valid = lengths[lengths >= min_len].index
    df = df[df["track_id"].isin(valid)].copy()
    return df


def add_kinematics(df: pd.DataFrame, pixels_per_meter: float = 20.0, fps: float = 23.98) -> pd.DataFrame:
    df["x_center"] = (df["bbox_xyxy"].apply(lambda b: b[0]) + df["bbox_xyxy"].apply(lambda b: b[2])) / 2.0
    df["y_center"] = (df["bbox_xyxy"].apply(lambda b: b[1]) + df["bbox_xyxy"].apply(lambda b: b[3])) / 2.0
    df["x_m"] = df["x_center"] / pixels_per_meter
    df["y_m"] = df["y_center"] / pixels_per_meter
    df["t_sec"] = df["frame_idx"] / fps

    velocities = {}
    window = 5
    for track_id, group in df.groupby("track_id"):
        group = group.sort_values("frame_idx")
        xs = group["x_m"].values
        ys = group["y_m"].values
        times = group["t_sec"].values

        vx_series = [0.0]
        vy_series = [0.0]
        for i in range(1, len(group)):
            dt = times[i] - times[i-1]
            vx = (xs[i] - xs[i-1]) / dt if dt > 0 else 0.0
            vy = (ys[i] - ys[i-1]) / dt if dt > 0 else 0.0
            vx_series.append(vx)
            vy_series.append(vy)

        def smooth(series, win=window):
            out = []
            for i in range(len(series)):
                start = max(0, i - win // 2)
                end = min(len(series), i + win // 2 + 1)
                out.append(np.mean(series[start:end]))
            return out

        vx_smooth = smooth(vx_series)
        vy_smooth = smooth(vy_series)

        for idx, (_, row) in enumerate(group.iterrows()):
            velocities[row.name] = (vx_smooth[idx], vy_smooth[idx])

    df["vx"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[0])
    df["vy"] = df.index.map(lambda i: velocities.get(i, (0.0, 0.0))[1])
    return df


def compute_ttc_events(df: pd.DataFrame, distance_threshold: float = 3.0,
                       closing_rate_threshold: float = 0.5,
                       ttc_threshold: float = 5.0) -> list[dict]:
    events = []
    for frame_idx, frame_df in df.groupby("frame_idx"):
        records = frame_df.to_dict("records")
        for i in range(len(records)):
            for j in range(i+1, len(records)):
                a, b = records[i], records[j]
                pos_a = np.array([a["x_m"], a["y_m"]])
                pos_b = np.array([b["x_m"], b["y_m"]])
                vel_a = np.array([a["vx"], a["vy"]])
                vel_b = np.array([b["vx"], b["vy"]])
                rel_pos = pos_b - pos_a
                rel_vel = vel_b - vel_a
                dist = np.linalg.norm(rel_pos)
                if dist < distance_threshold:
                    closing_rate = -np.dot(rel_pos, rel_vel)
                    rel_speed_sq = np.dot(rel_vel, rel_vel)
                    if rel_speed_sq > 1e-9 and closing_rate > closing_rate_threshold:
                        ttc = closing_rate / rel_speed_sq
                    else:
                        ttc = float("inf")
                    if np.isfinite(ttc) and 0 < ttc < ttc_threshold:
                        events.append({
                            "video_id": "sample_video",
                            "event_id": f"TTC_{a['track_id']}_{b['track_id']}_{frame_idx}",
                            "metric_name": "TTC",
                            "track_id_a": str(a["track_id"]),
                            "track_id_b": str(b["track_id"]),
                            "start_frame": int(frame_idx),
                            "end_frame": int(frame_idx),
                            "min_value": float(ttc),
                            "threshold": ttc_threshold,
                            "severity": "critical" if ttc < 1.5 else "non_critical",
                            "metadata": {"distance_m": float(dist), "closing_rate_mps": float(closing_rate)}
                        })
    return events


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--graphs_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/models")
    parser.add_argument("--window_size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=1.5)
    parser.add_argument("--pixels_per_meter", type=float, default=20.0)
    parser.add_argument("--fps", type=float, default=23.98)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    # Load and prepare tracks
    df = load_tracks(args.tracks)
    df = filter_tracks(df)
    df = add_kinematics(df, pixels_per_meter=args.pixels_per_meter, fps=args.fps)

    # Compute TTC events
    ttc_events = compute_ttc_events(df)

    # Build windows and labels
    frame_ttc = defaultdict(list)
    for ev in ttc_events:
        frame_ttc[ev["start_frame"]].append(ev["min_value"])

    window_ds = SpatioTemporalWindowDataset(
        graph_dir=args.graphs_dir,
        window_size=args.window_size,
        stride=args.stride,
    )

    labels = []
    for i in range(len(window_ds)):
        frames = window_ds[i].frame_ids.tolist()
        vals = []
        for fid in frames:
            vals.extend(frame_ttc.get(fid, []))
        min_ttc = min(vals) if vals else float("inf")
        labels.append(1 if min_ttc < args.threshold else 0)

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    # Split
    indices = list(range(len(window_ds)))
    np.random.seed(42)
    np.random.shuffle(indices)
    split = int(0.7 * len(indices))
    train_idx = indices[:split]
    val_idx = indices[split:]

    if not has_torch_geometric or len(window_ds) == 0:
        print("Torch Geometric not available or no windows found.")
        return

    # Model
    first = window_ds[0]
    model = build_model(in_channels=first.x.size(1), hidden_channels=32)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for idx in train_idx:
            data = window_ds[idx]
            optimizer.zero_grad()
            out = model(data)
            target = labels_tensor[idx].unsqueeze(0)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for idx in val_idx:
                data = window_ds[idx]
                out = model(data)
                pred = (torch.sigmoid(out) > 0.5).float()
                correct += (pred.item() == labels_tensor[idx].item())
                total += 1
        acc = correct / total if total else 0.0
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs} | Loss: {total_loss/len(train_idx):.4f} | Val Acc: {acc:.3f}")

    # Save model
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    torch.save(model.state_dict(), output_dir / "gcn_risk_filtered.pt")

    # Save metrics
    metrics = {
        "label_threshold": args.threshold,
        "train_size": len(train_idx),
        "val_size": len(val_idx),
        "best_val_accuracy": float(best_acc),
        "num_positive": int(sum(labels)),
        "num_negative": int(len(labels) - sum(labels)),
        "epochs": args.epochs,
        "ttc_events_count": len(ttc_events),
    }
    write_json(output_dir / "training_metrics_filtered.json", metrics)
    print(f"Model and metrics saved to {output_dir}")

if __name__ == "__main__":
    main()
