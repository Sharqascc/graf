from __future__ import annotations

from graf.data.graph_dataset import GraphSampleDataset
from graf.evaluation.binary_metrics import binary_classification_metrics
from graf.graph.builders import build_graph_for_frame
from graf.models.gcn_risk import build_model


def build_demo_dataset() -> GraphSampleDataset:
    samples = [
        {
            "graph": build_graph_for_frame([
                {"track_id": 1, "frame_id": 1, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
                {"track_id": 2, "frame_id": 1, "actor_class": "two_wheeler", "x": 2.0, "y": 1.0, "vx": 1.0, "vy": 0.2},
            ], radius=5.0, actor_classes=["car", "two_wheeler"]),
            "label": 1,
        },
        {
            "graph": build_graph_for_frame([
                {"track_id": 3, "frame_id": 2, "actor_class": "car", "x": 0.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
                {"track_id": 4, "frame_id": 2, "actor_class": "pedestrian", "x": 9.0, "y": 1.0, "vx": 0.2, "vy": 0.0},
            ], radius=5.0, actor_classes=["car", "pedestrian"]),
            "label": 0,
        },
    ]
    return GraphSampleDataset(samples)


def main() -> None:
    dataset = build_demo_dataset()
    sample = dataset[0]
    in_channels = sample.x.size(-1) if hasattr(sample, "x") else len(sample["x"][0])
    model = build_model(in_channels=in_channels, hidden_channels=16)

    try:
        import torch
        from torch_geometric.loader import DataLoader
    except Exception:
        print("Torch/PyG not installed; cannot run evaluation.")
        return

    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    batch = next(iter(loader))
    model.eval()
    with torch.no_grad():
        logits = model(batch).cpu().tolist()
    y_true = batch.y.view(-1).cpu().tolist()
    metrics = binary_classification_metrics(y_true, logits)
    print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
