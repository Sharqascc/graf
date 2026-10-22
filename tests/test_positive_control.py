"""Structural checks for the item-2 positive control."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "positive_control.py"
_WF = _REPO / ".github" / "workflows" / "positive-control.yml"

pytestmark = pytest.mark.skipif(
    not _WF.exists(),
    reason="positive-control workflow file is not present in this "
    "environment (running inside the Docker image)",
)


def test_script_exists():
    assert _SCRIPT.exists()


def test_script_checks_frame_y_and_window_y():
    src = _SCRIPT.read_text()
    assert "_check_frame_y" in src
    assert "_check_data_y" in src
    assert "frame_y" in src
    assert "window_y" in src


def test_script_runs_sustained_agreement():
    src = _SCRIPT.read_text()
    assert "sustained_agreement" in src
    assert "apply_strategy" in src
    assert "_sustained_labels" in src


def test_script_trains_two_trees():
    src = _SCRIPT.read_text()
    assert "tree_on_n_critical" in src
    assert "tree_on_min_ttc" in src
    assert "DecisionTreeClassifier" in src


def test_workflow_dispatches_and_uploads():
    text = _WF.read_text()
    assert "workflow_dispatch" in text
    assert "positive_control.py" in text
    assert "upload-artifact" in text


def test_workflow_yaml_parses():
    data = yaml.safe_load(_WF.read_text())
    assert "jobs" in data
    assert "control" in data["jobs"]
