from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a toy model for GRAF.")
    parser.add_argument("--outdir", type=str, default="outputs/models", help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model_name": "toy_linear_model",
        "status": "trained",
        "epochs": 3,
        "metrics": {
            "loss": 0.123,
            "accuracy": 0.987,
        },
    }

    out_path = outdir / "toy_model.json"
    out_path.write_text(json.dumps(model_artifact, indent=2), encoding="utf-8")
    print(f"Wrote trained model artifact to {out_path}")


if __name__ == "__main__":
    main()
