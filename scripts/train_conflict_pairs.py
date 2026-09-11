"""CLI wrapper around graf.training.conflict_pairs.run_cross_validation."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graf.training.conflict_pairs import run_cross_validation


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tracks", required=True)
    p.add_argument("--graphs_dir", required=True)
    p.add_argument("--homography_config", required=True)
    p.add_argument("--output_dir", default="outputs/models_conflict_pairs")
    p.add_argument("--window_size", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--distance_threshold", type=float, default=5.0)
    p.add_argument("--min_interaction_frames", type=int, default=3)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--num_folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    run_cross_validation(
        tracks_path=args.tracks,
        graphs_dir=args.graphs_dir,
        homography_config=args.homography_config,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        distance_threshold=args.distance_threshold,
        min_interaction_frames=args.min_interaction_frames,
        epochs=args.epochs,
        num_folds=args.num_folds,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
