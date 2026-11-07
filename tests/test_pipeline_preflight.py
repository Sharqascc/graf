"""Tests for the multi-pipeline registry and preflight (item 14)."""

from __future__ import annotations

from pathlib import Path

import pytest

from graf.pipeline import (
    INSPECT_STEPS,
    PIPELINES,
    SCIENCE_STEPS,
    Context,
    PreflightIssue,
    format_preflight,
    get_step,
    preflight,
    select_steps,
)


def _ctx(root: Path) -> Context:
    return Context(site="test", video=None, root=root)


# --- registry ------------------------------------------------------


def test_science_pipeline_has_eight_steps():
    assert len(SCIENCE_STEPS) == 8


def test_inspect_pipeline_has_six_steps():
    assert len(INSPECT_STEPS) == 6


def test_inspect_step_names():
    expected = ["detect", "grid", "gate", "track", "pet", "visualize"]
    got = [s.name for s in INSPECT_STEPS]
    assert got == expected


def test_inspect_step_numbers():
    assert [s.number for s in INSPECT_STEPS] == [1, 2, 3, 4, 5, 6]


def test_pipelines_dict_has_both():
    assert set(PIPELINES.keys()) == {"science", "inspect"}


def test_select_steps_defaults_to_science():
    steps = select_steps("all")
    assert len(steps) == 8


def test_select_steps_inspect():
    steps = select_steps("all", pipeline="inspect")
    assert len(steps) == 6
    assert steps[0].name == "detect"
    assert steps[-1].name == "visualize"


def test_select_steps_inspect_one():
    steps = select_steps("2", pipeline="inspect")
    assert len(steps) == 1
    assert steps[0].name == "grid"


def test_select_steps_inspect_from():
    steps = select_steps("from-4", pipeline="inspect")
    assert [s.number for s in steps] == [4, 5, 6]


def test_get_step_science():
    assert get_step(3).name == "trajectory"


def test_get_step_inspect():
    assert get_step(2, pipeline="inspect").name == "grid"


def test_get_step_unknown_pipeline():
    with pytest.raises(KeyError):
        get_step(1, pipeline="bogus")


# --- preflight -----------------------------------------------------


def test_preflight_ready_for_science():
    """In the real repo, all 8 science scripts exist."""
    root = Path(__file__).resolve().parents[1]
    issues = preflight(SCIENCE_STEPS, _ctx(root))
    assert issues == []


def test_preflight_reports_missing_inspect_scripts(tmp_path):
    """With only detect+track scripts present, preflight reports 4."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "run_detection.py").write_text("# stub")
    (scripts / "run_tracking.py").write_text("# stub")
    issues = preflight(INSPECT_STEPS, _ctx(tmp_path))
    missing = {i.detail for i in issues}
    assert len(issues) == 4
    assert "scripts/draw_grid.py not found" in missing
    assert "scripts/draw_gates.py not found" in missing
    assert "scripts/compute_pet.py not found" in missing
    assert "scripts/render_results.py not found" in missing


def test_preflight_ignores_audit_inline(tmp_path):
    """Audit step has no script — must not be reported missing."""
    issues = preflight(SCIENCE_STEPS, _ctx(tmp_path))
    assert all(i.step != 8 for i in issues)


def test_preflight_issue_carries_step_identity():
    issue = PreflightIssue(step=2, step_name="grid", detail="missing")
    assert issue.step == 2
    assert issue.step_name == "grid"


def test_format_preflight_lists_ready_steps(tmp_path):
    """When no scripts exist, all 6 inspect steps are NOT READY."""
    issues = preflight(INSPECT_STEPS, _ctx(tmp_path))
    text = format_preflight(issues, INSPECT_STEPS)
    assert "0 of 6 step(s) ready" in text
    assert "NOT READY" in text


def test_format_preflight_all_ready():
    text = format_preflight([], INSPECT_STEPS)
    assert "6 of 6 step(s) ready" in text
    assert "NOT READY" not in text
