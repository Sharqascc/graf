from __future__ import annotations

import argparse
import sys
from pathlib import Path

from graf.utils.logger import get_logger
from graf.utils.pipeline_status import print_pipeline_status
from graf.utils.export_graph_samples import export_graph_samples

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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
