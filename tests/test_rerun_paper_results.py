"""Tests for scripts/rerun_paper_results.py.

Every committed docs/paper/*.json with a setup block must produce a
well-formed compare_baselines.py command, and that command must carry
--ttc-dcpa-threshold 1.5 (the B.1 gate). Runs without data files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from scripts import rerun_paper_results as rp

REPO = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO / "docs" / "paper"


def _setup_jsons() -> list[tuple[Path, dict]]:
    out = []
    for jf in sorted(PAPER_DIR.glob("*.json")):
        try:
            payload = json.loads(jf.read_text())
        except Exception:
            continue
        setup = payload.get("setup")
        if isinstance(setup, dict) and setup:
            out.append((jf, setup))
    return out


_SETUPS = _setup_jsons()
_IDS = [jf.name for jf, _ in _SETUPS]


def test_at_least_one_artifact_has_setup():
    assert _SETUPS, (
        f"no artifacts under {PAPER_DIR} carry a setup block; the re-run "
        "helper has nothing to reconstruct"
    )


@pytest.mark.parametrize("jf,setup", _SETUPS, ids=_IDS)
def test_every_setup_yields_dcpa_1_5(jf, setup):
    args, _, _ = rp.build_args(setup)
    assert "--ttc-dcpa-threshold" in args, (
        f"{jf.name}: reconstructed command omits --ttc-dcpa-threshold (B.1 regression)"
    )
    idx = args.index("--ttc-dcpa-threshold")
    assert args[idx + 1] == "1.5"


def test_setup_without_dcpa_gets_15():
    args, _, _ = rp.build_args(
        {
            "tracks": "/tmp/x.jsonl",
            "graphs_dir": "/tmp/g",
            "pixels_per_meter": 100.0,
        }
    )
    idx = args.index("--ttc-dcpa-threshold")
    assert args[idx + 1] == "1.5"
    assert args.count("--ttc-dcpa-threshold") == 1


def test_bool_flag_calibrate():
    on_args, _, _ = rp.build_args({"calibrate_threshold": True})
    assert "--calibrate-threshold" in on_args
    off_args, _, _ = rp.build_args({"calibrate_threshold": False})
    assert "--calibrate-threshold" not in off_args


def test_unknown_key_warns():
    _, warns, _ = rp.build_args({"totally_made_up": 42})
    assert any("totally_made_up" in w for w in warns)


def test_missing_homography_flagged():
    _, _, missing = rp.build_args(
        {
            "tracks": "/tmp/x.jsonl",
            "graphs_dir": "/tmp/g",
        }
    )
    assert any("homography" in m or "pixels_per_meter" in m for m in missing)


def test_setup_with_pixels_per_meter_ok():
    _, _, missing = rp.build_args(
        {
            "tracks": "/tmp/x.jsonl",
            "graphs_dir": "/tmp/g",
            "pixels_per_meter": 100.0,
        }
    )
    assert not any("homography" in m or "pixels_per_meter" in m for m in missing)


def test_setup_with_homography_ok():
    _, _, missing = rp.build_args(
        {
            "tracks": "/tmp/x.jsonl",
            "graphs_dir": "/tmp/g",
            "homography_config": "/tmp/h.yaml",
        }
    )
    assert not any("homography" in m or "pixels_per_meter" in m for m in missing)
