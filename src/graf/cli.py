from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graf.training.conflict_pairs import run_cross_validation
from graf.utils.export_graph_samples import export_graph_samples
from graf.utils.logger import get_logger
from graf.utils.pipeline_status import print_pipeline_status

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graf",
        description="GRAF: graph-based surrogate safety analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="Show pipeline status")
    status.add_argument("--root", type=str, default=".", help="Repository root")
    status.add_argument("--depth", type=int, default=4, help="Tree print depth")

    demo = subparsers.add_parser("demo-graphs", help="Export a toy graph sample")
    demo.add_argument(
        "--outdir", type=str, default="outputs/graphs", help="Output directory"
    )

    train_cp = subparsers.add_parser(
        "train-conflict-pairs",
        help="Train a GCN risk classifier on conflict-pair labels (CV)",
    )
    train_cp.add_argument("--tracks", required=True, help="JSONL tracks file")
    train_cp.add_argument(
        "--graphs_dir", required=True, help="Directory of .pt graph files"
    )
    train_cp.add_argument(
        "--homography_config", required=True, help="YAML with homography matrix"
    )
    train_cp.add_argument(
        "--output_dir",
        default="outputs/models_conflict_pairs",
        help="Directory for metrics and checkpoints",
    )
    train_cp.add_argument("--window_size", type=int, default=5)
    train_cp.add_argument("--stride", type=int, default=2)
    train_cp.add_argument("--distance_threshold", type=float, default=5.0)
    train_cp.add_argument("--min_interaction_frames", type=int, default=3)
    train_cp.add_argument("--epochs", type=int, default=50)
    train_cp.add_argument("--num_folds", type=int, default=5)
    train_cp.add_argument("--seed", type=int, default=42)

    return parser


def run_status(root: str, depth: int = 4) -> int:
    repo_root = Path(root).resolve()
    try:
        print_pipeline_status(repo_root, depth=depth)
        return 0
    except Exception as e:
        logger.error("Pipeline status failed: %s", e)
        return 1


def run_demo_graphs(outdir: str) -> int:
    try:
        out_path = export_graph_samples(outdir)
        print(f"Wrote {out_path}")
        return 0
    except Exception as e:
        logger.error("Export graph samples failed: %s", e)
        return 1


def run_train_conflict_pairs(
    tracks: str,
    graphs_dir: str,
    homography_config: str,
    output_dir: str,
    window_size: int,
    stride: int,
    distance_threshold: float,
    min_interaction_frames: int,
    epochs: int,
    num_folds: int,
    seed: int,
) -> int:
    try:
        run_cross_validation(
            tracks_path=tracks,
            graphs_dir=graphs_dir,
            homography_config=homography_config,
            output_dir=output_dir,
            window_size=window_size,
            stride=stride,
            distance_threshold=distance_threshold,
            min_interaction_frames=min_interaction_frames,
            epochs=epochs,
            num_folds=num_folds,
            seed=seed,
        )
    except Exception as e:
        logger.error("Train conflict-pairs failed: %s", e)
        return 1
    print(f"Wrote CV metrics to {Path(output_dir) / 'metrics_cv.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    # Temporary workaround for JSON config invocation; can be removed later
    if argv and argv[0].startswith("/") and argv[0].endswith(".json"):
        argv = []

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        return run_status(args.root, args.depth)
    if args.command == "demo-graphs":
        return run_demo_graphs(args.outdir)
    if args.command == "train-conflict-pairs":
        return run_train_conflict_pairs(
            tracks=args.tracks,
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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
