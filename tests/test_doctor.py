"""Tests for graf.doctor."""

from __future__ import annotations

from pathlib import Path

from graf.doctor import (
    Check,
    check_detectors,
    check_dirs,
    check_python,
    check_recipes,
    run_checks,
)


def test_check_python_returns_check() -> None:
    c = check_python()
    assert isinstance(c, Check)
    assert c.name == "Python version"
    assert c.ok is True  # we require >=3.10; tests run on newer


def test_check_dirs_reports_missing(tmp_path: Path) -> None:
    checks = check_dirs(tmp_path)
    names = [c.name for c in checks]
    assert any("data/raw" in n for n in names)
    assert all(not c.ok for c in checks)  # empty tmp_path has nothing


def test_check_dirs_reports_present(tmp_path: Path) -> None:
    for rel in [
        "data",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "outputs",
        "configs",
        "configs/recipes",
    ]:
        (tmp_path / rel).mkdir(parents=True, exist_ok=True)
    checks = check_dirs(tmp_path)
    assert all(c.ok for c in checks)


def test_check_detectors_missing(tmp_path: Path) -> None:
    c = check_detectors(tmp_path)
    assert not c.ok
    assert "fetch-detector" in c.hint


def test_check_detectors_present(tmp_path: Path) -> None:
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "m.pt").write_bytes(b"x")
    c = check_detectors(tmp_path)
    assert c.ok


def test_check_recipes_missing(tmp_path: Path) -> None:
    c = check_recipes(tmp_path)
    assert not c.ok


def test_check_recipes_present(tmp_path: Path) -> None:
    (tmp_path / "configs" / "recipes").mkdir(parents=True)
    (tmp_path / "configs" / "recipes" / "r.yaml").write_text("name: x\nsteps: []\n")
    c = check_recipes(tmp_path)
    assert c.ok


def test_run_checks_returns_int(tmp_path: Path, capsys) -> None:
    rc = run_checks(tmp_path)
    assert rc in (0, 1)
    out = capsys.readouterr().out
    assert "GRAF environment check" in out
