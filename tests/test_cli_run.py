"""Tests for graf.pipeline (the `graf run` numbered registry)."""

from __future__ import annotations

from pathlib import Path

import pytest

from graf.pipeline import (
    STEPS,
    Command,
    Context,
    format_command,
    format_listing,
    get_step,
    select_steps,
)


def _ctx(site: str = "test_site") -> Context:
    return Context(site=site, video=None, root=Path("/repo"))


def test_eight_steps():
    assert len(STEPS) == 8


def test_step_numbers_and_names():
    expected = [
        (1, "detect"),
        (2, "track"),
        (3, "trajectory"),
        (4, "graph"),
        (5, "window"),
        (6, "train"),
        (7, "compare"),
        (8, "audit"),
    ]
    got = [(s.number, s.name) for s in STEPS]
    assert got == expected


def test_select_all():
    steps = select_steps("all")
    assert len(steps) == 8
    assert [s.number for s in steps] == list(range(1, 9))


def test_select_one():
    steps = select_steps("3")
    assert len(steps) == 1
    assert steps[0].number == 3
    assert steps[0].name == "trajectory"


def test_select_from():
    steps = select_steps("from-5")
    assert [s.number for s in steps] == [5, 6, 7, 8]


def test_select_bad_target():
    with pytest.raises(ValueError):
        select_steps("bogus")


def test_select_out_of_range():
    with pytest.raises(ValueError):
        select_steps("9")


def test_get_step():
    assert get_step(1).name == "detect"


def test_get_step_missing():
    with pytest.raises(ValueError):
        get_step(99)


def test_format_listing_lists_all_names():
    text = format_listing()
    for name in (
        "detect",
        "track",
        "trajectory",
        "graph",
        "window",
        "train",
        "compare",
        "audit",
    ):
        assert name in text


def test_step1_paths():
    ctx = _ctx("sci")
    cmds = STEPS[0].build_commands(ctx)
    assert len(cmds) == 1
    joined = " ".join(cmds[0].args)
    assert "/data/interim/frames/sci/sci" in joined
    assert "/data/interim/detections/sci" in joined


def test_step1_detection_config_falls_back():
    ctx = _ctx("no_such_site")
    cmds = STEPS[0].build_commands(ctx)
    joined = " ".join(cmds[0].args)
    assert "configs/detection/uvh26.yaml" in joined


def test_step8_uses_inline_audit():
    ctx = _ctx("test_site")
    cmds = STEPS[7].build_commands(ctx)
    assert len(cmds) == 1
    assert cmds[0].script == "_audit_inline"


def test_format_command_script():
    cmd = Command(script="run_detection.py", args=["--frames_dir", "x"])
    got = format_command(cmd)
    assert got == "python scripts/run_detection.py --frames_dir x"


def test_format_command_inline():
    cmd = Command(script="_audit_inline", args=["out.json"])
    assert format_command(cmd) == "graf audit out.json"
