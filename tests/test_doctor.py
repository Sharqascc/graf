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


# ------------------------------------------------------------------
# Coverage additions for doctor.py
# ------------------------------------------------------------------


def test_check_packages_reports_import_error(monkeypatch):
    """A missing optional package is reported as a failed check."""
    import importlib

    from graf import doctor as doc_mod

    real = importlib.import_module

    def fake_import(name):
        if name == "sklearn":
            raise ImportError("nope")
        return real(name)

    monkeypatch.setattr("graf.doctor.importlib.import_module", fake_import)
    checks = doc_mod.check_packages()
    skl = next(c for c in checks if "scikit-learn" in c.name)
    assert not skl.ok


def test_check_detectors_no_pt_files(tmp_path):
    from graf import doctor as doc_mod

    (tmp_path / "data" / "models").mkdir(parents=True)
    c = doc_mod.check_detectors(tmp_path)
    assert not c.ok
    assert "no .pt files" in c.detail


def test_check_recipes_no_yaml(tmp_path):
    from graf import doctor as doc_mod

    (tmp_path / "configs" / "recipes").mkdir(parents=True)
    c = doc_mod.check_recipes(tmp_path)
    assert not c.ok
    assert "no .yaml" in c.detail


def test_run_checks_default_root(monkeypatch):
    """run_checks(None) resolves the repo root and returns 0 on success."""
    from graf import doctor as doc_mod

    def pass_one(*a, **k):
        return doc_mod.Check("x", True, "ok")

    def pass_list(*a, **k):
        return [doc_mod.Check("x", True, "ok")]

    monkeypatch.setattr("graf.doctor.check_python", pass_one)
    monkeypatch.setattr("graf.doctor.check_packages", pass_list)
    monkeypatch.setattr("graf.doctor.check_dirs", pass_list)
    monkeypatch.setattr("graf.doctor.check_detectors", pass_one)
    monkeypatch.setattr("graf.doctor.check_recipes", pass_one)
    rc = doc_mod.run_checks()
    assert rc == 0


def test_run_checks_returns_one_when_failures(tmp_path, capsys):
    from graf import doctor as doc_mod

    rc = doc_mod.run_checks(tmp_path)
    assert rc == 1
    assert "Run `graf fetch-detector`" in capsys.readouterr().out
