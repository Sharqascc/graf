from __future__ import annotations

import argparse
import math
from pathlib import Path

from graf.data.graph_dataset import GraphSampleDataset
from graf.evaluation.binary_metrics import binary_classification_metrics
from graf.models.gcn_risk import build_model
from graf.utils.io import ensure_dir, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GCN risk checkpoint.")
    parser.add_argument("--graphs", type=str, required=True, help="Path to graph samples JSONL.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint.")
    parser.add_argument("--outdir", type=str, default="outputs/eval/gcn_risk", help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("Torch is required for evaluation.") from exc

    dataset = GraphSampleDataset.from_jsonl(args.graphs)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty.")

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = checkpoint.get("config", {})
    hidden_channels = int(config.get("hidden_channels", 64))

    sample = dataset[0]
    in_channels = sample.x.size(-1)
    model = build_model(in_channels=in_channels, hidden_channels=hidden_channels)
    if not hasattr(model, "load_state_dict"):
        raise RuntimeError("Loaded fallback non-torch model. Torch/PyG install is incomplete.")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows = []
    logits = []
    y_true = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        with torch.no_grad():
            out = model(item)
        logit = float(out.detach().cpu().view(-1)[0].item())
        prob = 1.0 / (1.0 + math.exp(-logit))

        sample_meta = dataset.samples[idx]
        label = int(sample_meta.get("label", 0))

        rows.append({
            "sample_id": sample_meta.get("sample_id", str(idx)),
            "site_id": sample_meta.get("site_id", ""),
            "video_id": sample_meta.get("video_id", ""),
            "window_id": sample_meta.get("window_id", ""),
            "label": label,
            "logit": logit,
            "prob": prob,
            "pred_label": int(prob >= 0.5),
        })
        logits.append(logit)
        y_true.append(label)

    metrics = binary_classification_metrics(y_true, logits)

    out_dir = ensure_dir(args.outdir)
    write_json(out_dir / "metrics.json", metrics)
    write_jsonl(out_dir / "predictions.jsonl", rows)

    print(f"saved_eval_dir={out_dir}")
    print({k: round(v, 4) for k, v in metrics.items()})


if __name__ == "__main__":
    main()
