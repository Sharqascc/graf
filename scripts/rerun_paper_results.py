"""Reconstruct and run the compare_baselines.py command for every paper artifact.

Every docs/paper/*.json file that has a ``setup`` block embeds the flags
that produced it. This script reads those blocks, emits the exact command
that would reproduce each artifact under B.1 (always injecting
``--ttc-dcpa-threshold 1.5``), and optionally runs them.

Setup blocks do not record ``--homography_config`` or ``--pixels_per_meter``.
Supply one of:

    --with-pixels-per-meter 100      # scale-only, e.g. VNTraffic at 100 px/m
    --with-homography path/to.yaml

Without one of those, every command is marked INCOMPLETE.

Examples:

    # Dry-run: print every command, flag incomplete artifacts
    python scripts/rerun_paper_results.py --with-pixels-per-meter 100 --check

    # Actually run
    python scripts/rerun_paper_results.py --with-pixels-per-meter 100 --run

Outputs land in ``outputs/rerun_b1/<artifact_stem>/`` so artifacts do not
overwrite each other.
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Setup key -> CLI flag.
KEY_TO_FLAG = {
    "tracks": "--tracks",
    "graphs_dir": "--graphs_dir",
    "homography_config": "--homography_config",
    "pixels_per_meter": "--pixels_per_meter",
    "window_size": "--window_size",
    "stride": "--stride",
    "label_source": "--label-source",
    "ttc_threshold_seconds": "--ttc-threshold-seconds",
    "ttc_distance_threshold": "--ttc-distance-threshold",
    "ttc_dcpa_threshold": "--ttc-dcpa-threshold",
    "split": "--split",
    "num_folds": "--num_folds",
    "epochs": "--epochs",
    "batch_size": "--batch_size",
    "lr": "--lr",
    "pos_weight": "--pos_weight",
    "seed": "--seed",
    "label_strategy": "--label-strategy",
    "run_length": "--run-length",
    "fraction_threshold": "--fraction-threshold",
    "calibration_objective": "--calibration-objective",
    "models": "--models",
}

BOOL_KEY_TO_FLAG = {
    "calibrate_threshold": "--calibrate-threshold",
}

NON_CLI_KEYS = {
    "num_windows",
    "duration_seconds",
    "elapsed_seconds",
    "purge_gap_frames",  # recorded but has no CLI flag
}

# Every re-run must use B.1. Injected if not already present.
FORCED_FLAGS = {"--ttc-dcpa-threshold": "1.5"}


def build_args(setup: dict) -> tuple[list[str], list[str], list[str]]:
    """Return (args, warnings, missing_required) for one setup block."""
    args: list[str] = []
    warnings: list[str] = []
    missing: list[str] = []

    for key, val in setup.items():
        if key in NON_CLI_KEYS:
            continue
        if key in BOOL_KEY_TO_FLAG:
            if val:
                args.append(BOOL_KEY_TO_FLAG[key])
            continue
        if key in KEY_TO_FLAG:
            if key == "tracks" and isinstance(val, str) and not val.endswith(
                (".jsonl", ".json")
            ):
                warnings.append(f"tracks is not a file path: {val!r}")
                missing.append("--tracks")
                continue
            args.extend([KEY_TO_FLAG[key], str(val)])
            continue
        warnings.append(f"unknown setup key {key!r}={val!r}")

    has_homography = "homography_config" in setup or "pixels_per_meter" in setup
    if not has_homography:
        missing.append("--homography_config or --pixels_per_meter")

    for flag, val in FORCED_FLAGS.items():
        if flag not in args:
            args.extend([flag, val])

    return args, warnings, missing


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--docs-dir", default=str(REPO / "docs" / "paper"))
    p.add_argument(
        "--with-pixels-per-meter",
        type=float,
        default=None,
        help="Add --pixels_per_meter N to any command that lacks homography.",
    )
    p.add_argument(
        "--with-homography",
        default=None,
        help="Add --homography_config PATH to any command that lacks homography.",
    )
    p.add_argument("--only", default=None,
                   help="Only process artifacts whose stem contains this string.")
    p.add_argument("--run", action="store_true",
                   help="Execute the commands (default: print only).")
    p.add_argument("--check", action="store_true",
                   help="Exit nonzero if any command is incomplete.")
    args = p.parse_args(argv)

    if args.with_pixels_per_meter is not None and args.with_homography is not None:
        print("error: choose at most one of --with-pixels-per-meter, "
              "--with-homography", file=sys.stderr)
        return 2

    docs = Path(args.docs_dir)
    artifacts = sorted(docs.glob("*.json"))
    if not artifacts:
        print(f"error: no artifacts under {docs}", file=sys.stderr)
        return 2

    n_ok = n_incomplete = 0
    incomplete_list: list[str] = []

    for jf in artifacts:
        if args.only and args.only not in jf.stem:
            continue
        try:
            payload = json.loads(jf.read_text())
        except Exception as e:
            print(f"[skip] {jf.name}: parse error: {e}")
            continue
        setup = payload.get("setup")
        if not isinstance(setup, dict) or not setup:
            print(f"[skip] {jf.name}: no setup block")
            continue

        cmd_args, warns, missing = build_args(setup)

        has_h = ("--homography_config" in cmd_args
                 or "--pixels_per_meter" in cmd_args)
        if not has_h and args.with_pixels_per_meter is not None:
            cmd_args.extend(["--pixels_per_meter",
                             str(args.with_pixels_per_meter)])
            missing = [m for m in missing if "homography" not in m
                       and "pixels_per_meter" not in m]
        elif not has_h and args.with_homography is not None:
            cmd_args.extend(["--homography_config", args.with_homography])
            missing = [m for m in missing if "homography" not in m
                       and "pixels_per_meter" not in m]

        cmd_args.extend([
            "--output_dir",
            str(REPO / "outputs" / "rerun_b1" / jf.stem),
        ])

        cmd = [sys.executable,
               str(REPO / "scripts" / "compare_baselines.py")] + cmd_args

        print(f"\n=== {jf.name} ===")
        print(shlex.join(cmd))
        for w in warns:
            print(f"  warn: {w}")
        if missing:
            n_incomplete += 1
            incomplete_list.append(jf.name)
            print(f"  INCOMPLETE: {', '.join(missing)}")
        else:
            n_ok += 1

        if args.run:
            if missing:
                print("  skip: incomplete")
                continue
            print("  running...")
            r = subprocess.run(cmd, cwd=REPO)
            print(f"  exit: {r.returncode}")
            if r.returncode != 0:
                n_incomplete += 1
                incomplete_list.append(jf.name + " (nonzero exit)")

    print(f"\nsummary: {n_ok} complete, {n_incomplete} incomplete")
    if incomplete_list:
        print(f"incomplete: {', '.join(incomplete_list)}")
    if args.check and n_incomplete:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
