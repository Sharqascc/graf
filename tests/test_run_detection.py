"""Tests for scripts/run_detection.py config handling.

Only the config-loading and resolution logic is tested here — the
detection loop itself requires ultralytics (an optional extra) and is
covered by the smoke-test run in studies/.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_detection as script_mod

REPO = Path(__file__).resolve().parents[1]


def _write_cfg(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_load_config_reads_yaml(tmp_path):
    p = _write_cfg(tmp_path / "c.yaml", {"weights": "y.pt", "imgsz": 960})
    cfg = script_mod.load_config(p)
    assert cfg["weights"] == "y.pt"
    assert cfg["imgsz"] == 960


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.load_config(tmp_path / "nope.yaml")


def test_load_config_empty_yaml_returns_empty_dict(tmp_path):
    p = tmp_path / "empty.yaml"
    p.write_text("", encoding="utf-8")
    assert script_mod.load_config(p) == {}


def test_load_config_non_mapping_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        script_mod.load_config(p)


def test_resolve_cli_wins():
    cfg = {"imgsz": 960}
    assert script_mod.resolve(cfg, "imgsz", 640, 320) == 640


def test_resolve_config_wins_when_no_cli():
    cfg = {"imgsz": 960}
    assert script_mod.resolve(cfg, "imgsz", None, 320) == 960


def test_resolve_default_when_neither():
    assert script_mod.resolve({}, "imgsz", None, 320) == 320


def test_resolve_null_in_config_falls_back_to_default():
    cfg = {"weights": None}
    assert script_mod.resolve(cfg, "weights", None, "yolov8n.pt") == "yolov8n.pt"


def test_default_config_exists_and_parses():
    cfg = script_mod.load_config(script_mod._DEFAULT_CONFIG)
    assert cfg["weights"] == "yolov8n.pt"
    assert cfg["imgsz"] == 1280


def test_script_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_detection.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--frames_dir",
        "--output_dir",
        "--config",
        "--model",
        "--stride",
        "--imgsz",
        "--conf",
        "--iou",
    ):
        assert flag in r.stdout
