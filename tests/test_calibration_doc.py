"""Structural checks on docs/paper/calibration_objective_post_purge.md.

The doc reports numbers that live in a sibling comparison.json when one
is checked in. These tests verify the doc is present, contains the
required sections and the expected model names, and — when the source
JSON is available — that the headline numbers agree.

The tests do not re-run the calibration. They catch the case where the
doc or the JSON is updated but not the other.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_DOC = _REPO / "docs" / "paper" / "calibration_objective_post_purge.md"
_PRIMARY = _REPO / "docs" / "paper" / "baselines_sustained_r5_post_purge.md"


def test_doc_exists():
    assert _DOC.exists(), f"missing doc: {_DOC}"


def test_doc_non_empty():
    text = _DOC.read_text()
    assert len(text) > 1500, f"doc suspiciously short: {len(text)} bytes"


def test_doc_has_required_sections():
    text = _DOC.read_text()
    for heading in (
        "## Why this doc exists",
        "## Setup",
        "## Results",
        "## Finding 1",
        "## Finding 2",
        "## Finding 3",
        "## The mechanism",
        "## How to reproduce",
        "## Cross-reference",
    ):
        assert heading in text, f"missing section: {heading!r}"


def test_doc_reports_both_objectives():
    text = _DOC.read_text()
    assert "--calibration-objective accuracy" in text
    assert "--calibration-objective youden" in text
    assert "Objective = accuracy" in text
    assert "Objective = youden" in text


def test_doc_reports_all_four_models():
    text = _DOC.read_text()
    for model in ("majority", "logreg", "rf", "single_feature"):
        assert model in text, f"missing model row: {model!r}"


def test_doc_reports_the_declared_setup_numbers():
    """The declared setup produced these calibrated accuracies."""
    text = _DOC.read_text()
    for value in (
        "0.703",  # rf, accuracy objective
        "0.661",  # logreg, accuracy objective
        "0.628",  # single_feature, accuracy objective
        "0.604",  # rf, youden objective
    ):
        assert value in text, f"missing calibrated-accuracy number: {value!r}"


def test_doc_cross_references_primary_result():
    text = _DOC.read_text()
    assert "baselines_sustained_r5_post_purge.md" in text
    assert _PRIMARY.exists()


def test_doc_cross_references_preregistration():
    text = _DOC.read_text()
    assert "preregistration_sustained_r5.md" in text


def test_doc_is_addendum_not_replacement():
    """The doc must say it does not change the primary result."""
    text = _DOC.read_text().lower()
    assert "addendum" in text
    assert "unchanged" in text or "does not change" in text


@pytest.mark.skipif(
    not (_REPO / "docs" / "paper" / "baselines_sustained_r5_post_purge.json").exists(),
    reason="primary comparison.json not checked in",
)
def test_doc_majority_number_matches_primary_json():
    """If the primary JSON is present, the majority baseline it records
    must match the majority number this addendum reports."""
    primary = _REPO / "docs" / "paper" / "baselines_sustained_r5_post_purge.json"
    payload = json.loads(primary.read_text())
    majority = payload["labels"]["majority_baseline"]
    majority_str = f"{majority:.3f}"
    text = _DOC.read_text()
    assert majority_str in text, (
        f"doc does not mention primary JSON majority baseline {majority_str!r}"
    )
