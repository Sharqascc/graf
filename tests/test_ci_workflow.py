"""Structural tests for the docs-only optimization in ci.yml."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[1]
_CI = _REPO / ".github" / "workflows" / "ci.yml"


def _jobs():
    return yaml.safe_load(_CI.read_text())["jobs"]


def test_ci_yaml_exists_and_parses():
    assert _CI.exists()
    data = yaml.safe_load(_CI.read_text())
    assert "jobs" in data


def test_changes_job_exists_and_outputs_code():
    jobs = _jobs()
    assert "changes" in jobs
    assert "outputs" in jobs["changes"]
    assert "code" in jobs["changes"]["outputs"]


def test_heavy_jobs_gated_on_changes():
    jobs = _jobs()
    for name in ("invariants", "docker"):
        job = jobs[name]
        assert job.get("needs") == "changes", f"{name} needs changes"
        assert "needs.changes.outputs.code" in job.get("if", ""), (
            f"{name} if-guard missing"
        )


def test_filters_cover_code_paths():
    jobs = _jobs()
    filters = str(jobs["changes"]["steps"])
    for path in (
        "src/",
        "tests/",
        "scripts/",
        "Dockerfile",
        "pyproject.toml",
        "requirements/",
    ):
        assert path in filters, f"filter for {path!r} missing"


def test_required_jobs_not_gated():
    jobs = _jobs()
    for name in ("lint", "unit-tests", "property-tests"):
        needs = str(jobs[name].get("needs", ""))
        assert "changes" not in needs, (
            f"{name} is a required check and must not be gated"
        )
