"""CLI wrapper around graf.cli.run_train_conflict_pairs.

Kept as a standalone entrypoint for script-style invocations (e.g. the
documented example in docs/experiment_results.md). Flags are defined in
one place — ``graf.cli.build_parser()`` — so the script and the
``graf train-conflict-pairs`` subcommand cannot drift.

Prefer ``graf train-conflict-pairs ...`` for new work; this script exists
for backwards compatibility with existing docs and CI invocations.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from graf.cli import build_parser, run_train_conflict_pairs


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = build_parser().parse_args(["train-conflict-pairs", *argv])
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


if __name__ == "__main__":
    raise SystemExit(main())
