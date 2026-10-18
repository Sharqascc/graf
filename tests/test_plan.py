"""Guards on PLAN.md: a done item must have real evidence."""

from __future__ import annotations

import re
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PLAN = _REPO / "PLAN.md"
_AGENTS = _REPO / "AGENTS.md"


def _items(text):
    # Non-capturing group: re.split returns only the blocks, not the
    # numbers. A capturing group would interleave them and put the
    # digit string into the blocks list.
    return re.split(r"^### (?:\d+)\. ", text, flags=re.MULTILINE)[1:]


def test_plan_exists():
    assert _PLAN.exists()


def test_plan_items_are_sequential():
    text = _PLAN.read_text()
    nums = [int(n) for n in re.findall(r"^### (\d+)\. ", text, flags=re.MULTILINE)]
    assert nums == list(range(1, len(nums) + 1)), nums


def test_every_item_has_status():
    blocks = _items(_PLAN.read_text())
    for i, block in enumerate(blocks, 1):
        assert "status:" in block, f"item {i} has no status line"


def test_done_items_have_existing_evidence():
    blocks = _items(_PLAN.read_text())
    for i, block in enumerate(blocks, 1):
        s = re.search(r"^status:\s*(\w+)", block, re.MULTILINE)
        e = re.search(r"^evidence:\s*(\S+)", block, re.MULTILINE)
        if s and s.group(1) == "done":
            assert e, f"item {i} done but no evidence path"
            assert (_REPO / e.group(1)).exists(), f"item {i} evidence missing"


def test_current_focus_names_one_item():
    m = re.search(r"Current focus: item (\d+)", _PLAN.read_text())
    assert m, "PLAN.md must declare a current focus"


def test_agents_points_at_plan():
    assert _AGENTS.exists()
    assert "PLAN.md" in _AGENTS.read_text()
