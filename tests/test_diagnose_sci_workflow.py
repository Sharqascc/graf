"""Structural checks on .github/workflows/diagnose-sci.yml.

The workflow downloads an SCI pipeline artifact and runs
scripts/diagnose_tracking.py against it. These tests verify the
workflow file is present, dispatchable, references the expected
run id, and runs the correct script with the correct arguments.
"""

from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_WORKFLOW = _REPO / ".github" / "workflows" / "diagnose-sci.yml"


def test_workflow_exists():
    assert _WORKFLOW.exists(), f"missing workflow: {_WORKFLOW}"


def test_workflow_is_dispatchable():
    text = _WORKFLOW.read_text()
    assert "workflow_dispatch" in text
    assert "sci_run_id" in text


def test_workflow_defaults_to_stride_1_run():
    text = _WORKFLOW.read_text()
    assert "38048010561" in text, "stride-1 run id missing from default"


def test_workflow_downloads_by_run_id():
    text = _WORKFLOW.read_text()
    assert "download-artifact" in text
    assert "run-id" in text
    assert "github-token" in text


def test_workflow_runs_diagnose_tracking():
    text = _WORKFLOW.read_text()
    assert "diagnose_tracking.py" in text
    assert "--tracks" in text
    assert "--output_dir" in text


def test_workflow_uploads_diagnosis():
    text = _WORKFLOW.read_text()
    assert "upload-artifact" in text
    assert "diagnosis.json" in text
