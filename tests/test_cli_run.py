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


def test_run_pipeline_prints_preflight_table(tmp_path, monkeypatch, capsys):
    """On a clean run, the preflight table is printed before work."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_detection.py").write_text("")

    class _R:
        returncode = 0

    monkeypatch.setattr("graf.cli.subprocess.run", lambda *a, **kw: _R())

    from graf.cli import run_pipeline

    rc = run_pipeline("1", site="t", list_only=False, root=str(tmp_path))
    out = capsys.readouterr().out

    assert rc == 0
    assert "step        status" in out
    assert "1 of 1 step(s) ready" in out


def test_run_pipeline_prints_argv_per_command(tmp_path, monkeypatch, capsys):
    """The exact script and args are printed before the subprocess runs."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_detection.py").write_text("")

    class _R:
        returncode = 0

    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return _R()

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)

    from graf.cli import run_pipeline

    run_pipeline("1", site="t", list_only=False, root=str(tmp_path))
    out = capsys.readouterr().out

    assert "python scripts/run_detection.py" in out
    assert "--frames_dir" in out
    assert "--config" in out
    assert len(calls) == 1


def test_run_pipeline_writes_run_log(tmp_path, monkeypatch):
    """Every non-dry-run writes a timestamped log under outputs/."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_detection.py").write_text("")

    class _R:
        returncode = 0

    monkeypatch.setattr("graf.cli.subprocess.run", lambda *a, **kw: _R())

    from graf.cli import run_pipeline

    run_pipeline("1", site="t", list_only=False, root=str(tmp_path))

    logs = list((tmp_path / "outputs").glob("run_log_*.txt"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert "detect" in content
    assert "run_detection.py" in content
    assert "1 of 1 steps passed" in content


def test_run_pipeline_dry_run_does_not_execute(tmp_path, monkeypatch, capsys):
    """--dry-run prints the table, returns 0, runs no subprocess."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_detection.py").write_text("")

    called = []

    def fake_run(cmd, **kw):
        called.append(cmd)
        return type("R", (), {"returncode": 0})()

    monkeypatch.setattr("graf.cli.subprocess.run", fake_run)

    from graf.cli import run_pipeline

    rc = run_pipeline("1", site="t", list_only=False, root=str(tmp_path), dry_run=True)
    out = capsys.readouterr().out

    assert rc == 0
    assert "dry run: no commands executed" in out
    assert called == []
    assert not (tmp_path / "outputs").exists()


def test_run_pipeline_failure_stops_and_logs(tmp_path, monkeypatch):
    """A non-zero subprocess return stops execution and is logged."""
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "run_detection.py").write_text("")

    class _R:
        returncode = 7

    monkeypatch.setattr("graf.cli.subprocess.run", lambda *a, **kw: _R())

    from graf.cli import run_pipeline

    rc = run_pipeline("1", site="t", list_only=False, root=str(tmp_path))

    assert rc == 7
    logs = list((tmp_path / "outputs").glob("run_log_*.txt"))
    assert len(logs) == 1
    content = logs[0].read_text()
    assert "failed with exit code 7" in content
