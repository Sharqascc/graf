"""Structural checks on docs/sci_tracking_improvement_plan.md.

The plan is a pre-registration: it commits the sequence of SCI
tracking fixes before any is applied. These tests verify the plan
is present, has the required sections, that the phases are numbered
in order, and that the pre-committed decision rules are stated.
They do not enforce execution order; they prevent the plan from
being silently truncated after an unfavorable phase result.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PLAN = _REPO / "docs" / "sci_tracking_improvement_plan.md"


def test_plan_exists():
    assert _PLAN.exists(), f"missing plan: {_PLAN}"


def test_plan_declares_pre_registration():
    text = _PLAN.read_text()
    assert "pre-registration" in text.lower()


def test_plan_has_required_sections():
    text = _PLAN.read_text()
    for heading in (
        "## Why this document exists",
        "## Current state",
        "## The improvement menu",
        "## Decision rules",
        "## Phases",
        "## What this plan commits the analysis not to do",
        "## Cross-reference",
    ):
        assert heading in text, f"missing section: {heading!r}"


def test_phases_are_numbered_sequentially():
    text = _PLAN.read_text()
    found = re.findall(r"^### Phase (\d+)", text, re.MULTILINE)
    assert found == ["0", "1", "2", "3", "4"], f"phases: {found}"


def test_plan_has_decision_rules_for_each_transition():
    text = _PLAN.read_text()
    for transition in (
        "Phase 0 -> Phase 1",
        "Phase 1 -> Phase 2",
        "Phase 2 -> Phase 3",
        "Phase 3 -> Phase 4",
    ):
        assert transition in text, f"missing decision rule: {transition!r}"


def test_plan_lists_five_commitments():
    text = _PLAN.read_text()
    marker = "## What this plan commits the analysis not to do"
    section = text.split(marker, 1)[1]
    section = section.split("## Cross-reference", 1)[0]
    for n in range(1, 6):
        assert f"{n}." in section, f"commitment {n} missing"


def test_plan_cites_the_current_run_ids():
    text = _PLAN.read_text()
    assert "38045758114" in text, "stride-3 run id missing"
    assert "38048010561" in text, "stride-1 run id missing"


def test_plan_cross_references_preregistration():
    text = _PLAN.read_text()
    assert "preregistration_sustained_r5.md" in text
    assert "external_review_2026_09.md" in text


def test_plan_states_phase_1_thresholds():
    text = _PLAN.read_text()
    assert "2 metres" in text or "2 m" in text, (
        "Phase 1 must state the displacement threshold"
    )
