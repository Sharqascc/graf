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
    """Only the Phases section defines phases. The Decision rules
    section also uses ``### Phase N -> Phase M`` headings; scope the
    regex to the Phases section so it does not double-count."""
    text = _PLAN.read_text()
    phases_section = text.split("## Phases", 1)[1]
    phases_section = phases_section.split("## What this plan commits", 1)[0]
    found = re.findall(r"^### Phase (\d+) ", phases_section, re.MULTILINE)
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

def test_addendum_records_phase_1_2_result():
    """Addendum 1 must exist and cite both the baseline and the new run."""
    text = _PLAN.read_text()
    assert "Addendum 1" in text, "addendum 1 missing from plan"
    assert "38557140055" in text, "Phase 1+2 run id not cited"
    assert "38048010561" in text, "baseline run id not cited"


def test_addendum_confirms_phase_3_per_decision_rule():
    """The addendum must confirm Phase 3, not re-tune the gates."""
    text = _PLAN.read_text()
    marker = "## Addendum 1"
    section = text.split(marker, 1)[1]
    section = section.split("## Cross-reference", 1)[0]
    assert "Phase 3" in section, "addendum does not name the next phase"
    assert "65.9%" in section, "addendum does not cite the decision metric"
    # must explicitly disclaim re-tuning
    assert "not a re-tune" in section or "not a retune" in section

