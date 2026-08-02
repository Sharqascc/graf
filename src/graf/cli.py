from __future__ import annotations

import argparse
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="graf",
        description="GRAF: graph-based surrogate safety analysis pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    status = subparsers.add_parser("status", help="Show pipeline status")
    status.add_argument("--root", type=str, default=".", help="Repository root")

    demo = subparsers.add_parser("demo-graphs", help="Export a toy graph sample")
    demo.add_argument("--outdir", type=str, default="outputs/graphs", help="Output directory")

    return parser


def _run_pipeline_status(root: str) -> int:
    repo_root = Path(root).resolve()
    sys.path.insert(0, str(repo_root))
    from scripts.pipeline_status import main as pipeline_status_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ["pipeline_status.py", "--root", str(repo_root)]
        pipeline_status_main()
    finally:
        sys.argv = old_argv
    return 0


def _run_export_graph_samples(outdir: str) -> int:
    repo_root = Path.cwd().resolve()
    sys.path.insert(0, str(repo_root))
    from scripts.export_graph_samples import main as export_graph_samples_main

    old_argv = sys.argv[:]
    try:
        sys.argv = ["export_graph_samples.py", "--outdir", str(outdir)]
        export_graph_samples_main()
    finally:
        sys.argv = old_argv
    return 0


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    if argv and argv[0].startswith("/") and argv[0].endswith(".json"):
        argv = []

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "status":
        return _run_pipeline_status(args.root)
    if args.command == "demo-graphs":
        return _run_export_graph_samples(args.outdir)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
