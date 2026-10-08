from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from graf.doctor import run_checks as run_doctor_checks
from graf.reproduce import load_recipe, run_recipe
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

    reproduce = subparsers.add_parser(
        "reproduce",
        help="Run a recipe file (a sequence of pipeline steps)",
    )
    reproduce.add_argument("recipe", help="Path to a recipe YAML")
    reproduce.add_argument(
        "--dry-run",
        action="store_true",
        help="Show steps without executing them",
    )
    reproduce.add_argument(
        "--only",
        default=None,
        help="Run only the step with this name",
    )
    reproduce.add_argument(
        "--from-step",
        default=None,
        help="Start from the step with this name (inclusive)",
    )

    doctor = subparsers.add_parser(
        "doctor",
        help="Check environment, data, and detector weights",
    )
    doctor.add_argument(
        "--root",
        type=str,
        default=None,
        help="Repository root (defaults to the package's repo root)",
    )

    fetch_det = subparsers.add_parser(
        "fetch-detector",
        help="Download detector weights (UVH-26 variants)",
    )
    fetch_det.add_argument(
        "--model",
        default="YOLOv11-S",
        help="Which UVH-26 variant (default: YOLOv11-S)",
    )
    fetch_det.add_argument(
        "--output-dir",
        default="data/models",
        help="Where to write weights",
    )

    detect = subparsers.add_parser(
        "detect-video",
        help="Run a detector over a frames directory (no ground truth needed)",
    )
    detect.add_argument("--frames-dir", required=True)
    detect.add_argument("--model", required=True, help="Path to .pt weights")
    detect.add_argument("--output-dir", required=True)
    detect.add_argument("--samples", type=int, default=6)
    detect.add_argument("--conf", type=float, default=0.25)
    detect.add_argument("--imgsz", type=int, default=640)

    return parser


def _repo_root() -> Path:
    """Return the repository root: two levels above src/graf/."""
    return Path(__file__).resolve().parents[2]


def run_doctor(root: str | None = None) -> int:
    if root is None:
        root_path = _repo_root()
    else:
        root_path = Path(root).resolve()
    try:
        return run_doctor_checks(root_path)
    except Exception as e:
        logger.error("Doctor failed: %s", e)
        return 1


def run_fetch_detector(model: str, output_dir: str) -> int:
    script = _repo_root() / "scripts" / "fetch_detector.py"
    if not script.exists():
        logger.error("script not found: %s", script)
        return 1
    cmd = [sys.executable, str(script), "--model", model, "--output-dir", output_dir]
    try:
        return subprocess.run(cmd).returncode
    except Exception as e:
        logger.error("fetch-detector failed: %s", e)
        return 1


def run_detect_video(
    frames_dir: str,
    model: str,
    output_dir: str,
    samples: int,
    conf: float,
    imgsz: int,
) -> int:
    script = _repo_root() / "scripts" / "detect_video.py"
    if not script.exists():
        logger.error("script not found: %s", script)
        return 1
    cmd = [
        sys.executable,
        str(script),
        "--frames_dir",
        frames_dir,
        "--model",
        model,
        "--output_dir",
        output_dir,
        "--samples",
        str(samples),
        "--conf",
        str(conf),
        "--imgsz",
        str(imgsz),
    ]
    try:
        return subprocess.run(cmd).returncode
    except Exception as e:
        logger.error("detect-video failed: %s", e)
        return 1


def run_reproduce(
    recipe_path: str,
    dry_run: bool = False,
    only: str | None = None,
    from_step: str | None = None,
) -> int:
    try:
        recipe = load_recipe(recipe_path)
    except (FileNotFoundError, ValueError) as e:
        logger.error("Could not load recipe: %s", e)
        return 1
    return run_recipe(recipe, dry_run=dry_run, only=only, from_step=from_step)


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
    if args.command == "doctor":
        return run_doctor(args.root)
    if args.command == "fetch-detector":
        return run_fetch_detector(args.model, args.output_dir)
    if args.command == "detect-video":
        return run_detect_video(
            frames_dir=args.frames_dir,
            model=args.model,
            output_dir=args.output_dir,
            samples=args.samples,
            conf=args.conf,
            imgsz=args.imgsz,
        )
    if args.command == "reproduce":
        return run_reproduce(
            recipe_path=args.recipe,
            dry_run=args.dry_run,
            only=args.only,
            from_step=args.from_step,
        )
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
