from __future__ import annotations

import argparse
import math
from datetime import datetime, UTC
from pathlib import Path

from graf.data.graph_dataset import GraphSampleDataset
from graf.graph.pyg_export import to_pyg_data
from graf.evaluation.binary_metrics import binary_classification_metrics
from graf.models.gcn_risk import build_model
from graf.utils.io import ensure_dir, get_git_commit, snapshot_environment, write_json, write_jsonl
from graf.utils.seeds import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GCN risk model on graph JSONL samples.")
    parser.add_argument("--graphs", type=str, required=True, help="Path to graph samples JSONL.")
    parser.add_argument("--outdir", type=str, default="outputs/models/gcn_risk", help="Base output directory.")
    parser.add_argument("--group-key", type=str, default="site_id", help="Grouping key for leakage-safe splitting.")
    parser.add_argument("--val-frac", type=float, default=0.2, help="Validation fraction at group level.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--hidden-channels", type=int, default=64, help="Hidden channels for GCN.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Adam weight decay.")
    return parser.parse_args()


def _sample_meta(dataset: GraphSampleDataset, index: int) -> dict:
    sample = dataset.samples[index]
    return {
        "sample_id": sample.get("sample_id", str(index)),
        "site_id": sample.get("site_id", ""),
        "video_id": sample.get("video_id", ""),
        "window_id": sample.get("window_id", ""),
        "label": int(sample.get("label", 0)),
    }


def grouped_split_indices(samples: list[dict], group_key: str, val_frac: float, seed: int) -> tuple[list[int], list[int]]:
    groups = []
    for i, sample in enumerate(samples):
        group = sample.get(group_key)
        if group in (None, ""):
            group = sample.get("video_id") or sample.get("site_id") or f"ungrouped_{i}"
        groups.append(str(group))

    unique_groups = sorted(set(groups))
    if len(unique_groups) < 2:
        n = len(samples)
        split = max(1, int(round(n * (1.0 - val_frac))))
        split = min(split, max(n - 1, 1))
        return list(range(split)), list(range(split, n))

    try:
        from sklearn.model_selection import GroupShuffleSplit
        splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
        train_idx, val_idx = next(splitter.split(samples, groups=groups))
        return train_idx.tolist(), val_idx.tolist()
    except Exception:
        import random
        rng = random.Random(seed)
        shuffled = unique_groups[:]
        rng.shuffle(shuffled)
        n_val_groups = max(1, int(math.ceil(len(shuffled) * val_frac)))
        val_groups = set(shuffled[:n_val_groups])
        train_idx = [i for i, g in enumerate(groups) if g not in val_groups]
        val_idx = [i for i, g in enumerate(groups) if g in val_groups]
        if not train_idx or not val_idx:
            split = max(1, int(round(len(samples) * (1.0 - val_frac))))
            split = min(split, max(len(samples) - 1, 1))
            return list(range(split)), list(range(split, len(samples)))
        return train_idx, val_idx


def build_subset(dataset: GraphSampleDataset, indices: list[int]) -> GraphSampleDataset:
    return GraphSampleDataset([dataset.samples[i] for i in indices])


def evaluate(model, loader, device):
    import torch

    model.eval()
    losses = []
    logits_all = []
    y_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            logits = model(batch)
            y = batch.y.view(-1).float()
            loss = torch.nn.BCEWithLogitsLoss()(logits, y)
            losses.append(loss.detach().item())
            logits_all.extend(logits.detach().cpu().tolist())
            y_all.extend(y.detach().cpu().tolist())

    metrics = binary_classification_metrics(y_all, logits_all)
    metrics["loss"] = sum(losses) / max(len(losses), 1)
    return metrics, logits_all, y_all


def main() -> None:
    args = parse_args()
    set_global_seed(args.seed, deterministic=True)

    try:
        import torch
        from torch_geometric.loader import DataLoader
    except Exception as exc:
        raise RuntimeError("Torch and torch_geometric must be installed to train the GCN risk model.") from exc

    raw_dataset = GraphSampleDataset.from_jsonl(args.graphs)
    dataset = [to_pyg_data(sample) for sample in raw_dataset.samples]
    if len(dataset) < 2:
        raise RuntimeError("Need at least 2 graph samples to train/evaluate.")

    train_idx, val_idx = grouped_split_indices(dataset.samples, args.group_key, args.val_frac, args.seed)
    train_ds = build_subset(dataset, train_idx)
    val_ds = build_subset(dataset, val_idx)

    sample = train_ds[0]
    in_channels = sample.x.size(-1)

    model = build_model(in_channels=in_channels, hidden_channels=args.hidden_channels)
    if not hasattr(model, "parameters"):
        raise RuntimeError("Loaded fallback non-torch model. Torch/PyG install is incomplete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()

    commit = get_git_commit(Path(__file__).resolve().parents[1])
    run_name = f"{datetime.now(UTC).strftime('%Y-%m-%d_%H%M%S')}_{commit}"
    run_dir = ensure_dir(Path(args.outdir) / run_name)
    snapshot_environment(Path(__file__).resolve().parents[1], run_dir)

    config = vars(args).copy()
    config["train_size"] = len(train_ds)
    config["val_size"] = len(val_ds)
    config["device"] = str(device)
    config["git_commit"] = commit
    write_json(run_dir / "config.json", config)

    best_f1 = -1.0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            y = batch.y.view(-1).float()
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.detach().item())

        val_metrics, val_logits, val_targets = evaluate(model, val_loader, device)
        train_loss = sum(train_losses) / max(len(train_losses), 1)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"val_{k}": float(v) for k, v in val_metrics.items()},
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"val_f1={val_metrics.get('f1', 0.0):.4f} "
            f"val_acc={val_metrics.get('accuracy', 0.0):.4f}"
        )

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, run_dir / "last.pt")

        current_f1 = float(val_metrics.get("f1", 0.0))
        if current_f1 > best_f1:
            best_f1 = current_f1
            torch.save(checkpoint, run_dir / "best.pt")

    write_json(run_dir / "metrics.json", {
        "best_val_f1": best_f1,
        "history": history,
    })

    predictions = []
    model.eval()
    with torch.no_grad():
        for dataset_index in val_idx:
            item = dataset[dataset_index]
            item = item.to(device)
            logits = model(item)
            if hasattr(logits, "numel") and logits.numel() == 1:
                logit = float(logits.detach().cpu().view(-1)[0].item())
            else:
                logit = float(logits.detach().cpu().view(-1)[0].item())
            prob = 1.0 / (1.0 + math.exp(-logit))
            meta = _sample_meta(dataset, dataset_index)
            predictions.append({
                **meta,
                "split": "val",
                "logit": logit,
                "prob": prob,
                "pred_label": int(prob >= 0.5),
            })

    write_jsonl(run_dir / "predictions.jsonl", predictions)
    print(f"saved_run_dir={run_dir}")


if __name__ == "__main__":
    main()
