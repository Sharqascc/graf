"""Tests for the scripts/train_conflict_pairs.py compatibility shim.

The script delegates to graf.cli.run_train_conflict_pairs and reuses
graf.cli.build_parser for flag definitions. These tests verify the shim
stays wired correctly as both the CLI and the script evolve.

The full end-to-end invocation (with a synthetic dataset) is covered by
tests/test_end_to_end_synthetic.py — these tests are the cheap wiring
checks.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from scripts import train_conflict_pairs as script_mod

REPO = Path(__file__).resolve().parents[1]


def test_script_help_lists_flags():
    """--help works and lists the flags shared with the CLI subcommand."""
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "train_conflict_pairs.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--tracks",
        "--graphs_dir",
        "--homography_config",
        "--epochs",
        "--num_folds",
        "--seed",
    ):
        assert flag in r.stdout, flag


def test_script_delegates_to_cli_runner(monkeypatch):
    """main() calls graf.cli.run_train_conflict_pairs with parsed values."""
    called = {}

    def fake_runner(**kwargs):
        called.update(kwargs)
        return 0

    # Patch the reference imported into the script module
    monkeypatch.setattr(
        script_mod, "run_train_conflict_pairs", fake_runner, raising=True
    )

    rc = script_mod.main(
        [
            "--tracks",
            "t.jsonl",
            "--graphs_dir",
            "g",
            "--homography_config",
            "h.yaml",
            "--epochs",
            "7",
            "--num_folds",
            "3",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    assert called["tracks"] == "t.jsonl"
    assert called["epochs"] == 7
    assert called["num_folds"] == 3
    assert called["seed"] == 1
    # Defaults pass through
    assert called["window_size"] == 5
    assert called["stride"] == 2
    assert called["distance_threshold"] == 5.0
    assert called["min_interaction_frames"] == 3


def test_script_propagates_runner_return_code(monkeypatch):
    """If run_train_conflict_pairs returns non-zero, main() returns it."""
    monkeypatch.setattr(
        script_mod, "run_train_conflict_pairs", lambda **kw: 42, raising=True
    )
    rc = script_mod.main(
        [
            "--tracks",
            "t",
            "--graphs_dir",
            "g",
            "--homography_config",
            "h",
        ]
    )
    assert rc == 42


def test_script_missing_required_flag_exits_nonzero():
    """Missing --tracks produces a non-zero exit via SystemExit."""
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "train_conflict_pairs.py"),
            "--graphs_dir",
            "g",
            "--homography_config",
            "h",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode != 0
    assert "--tracks" in r.stderr or "--tracks" in r.stdout
