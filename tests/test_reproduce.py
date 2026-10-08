"""Tests for the recipe runner."""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from graf.reproduce import Recipe, load_recipe, run_recipe


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_load_recipe_minimal(tmp_path: Path) -> None:
    p = _write(tmp_path, {
        "name": "t",
        "steps": [{"name": "s1", "cmd": ["echo", "hi"]}],
    })
    r = load_recipe(p)
    assert r.name == "t"
    assert len(r.steps) == 1
    assert r.steps[0].kind == "cmd"
    assert r.steps[0].cmd == ["echo", "hi"]


def test_load_recipe_python_step(tmp_path: Path) -> None:
    p = _write(tmp_path, {
        "name": "t",
        "steps": [{"name": "s1", "python": "x = 1 + 1"}],
    })
    r = load_recipe(p)
    assert r.steps[0].kind == "python"
    assert r.steps[0].code == "x = 1 + 1"


def test_load_recipe_rejects_both_cmd_and_python(tmp_path: Path) -> None:
    p = _write(tmp_path, {
        "steps": [{"name": "s1", "cmd": ["echo"], "python": "x = 1"}],
    })
    with pytest.raises(ValueError, match="exactly one"):
        load_recipe(p)


def test_load_recipe_rejects_neither(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1"}]})
    with pytest.raises(ValueError, match="exactly one"):
        load_recipe(p)


def test_load_recipe_rejects_cmd_not_list(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": "echo hi"}]})
    with pytest.raises(ValueError, match="list of strings"):
        load_recipe(p)


def test_load_recipe_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/recipe.yaml")


def test_run_recipe_dry_run_does_not_execute(tmp_path: Path) -> None:
    p = _write(tmp_path, {
        "name": "t",
        "steps": [
            {"name": "s1", "cmd": ["false"]},  # would fail if executed
            {"name": "s2", "python": "raise SystemExit(1)"},
        ],
    })
    r = load_recipe(p)
    rc = run_recipe(r, dry_run=True)
    assert rc == 0


def test_run_recipe_only_selects_one_step(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "bad", "cmd": ["false"]},
                {"name": "good", "cmd": ["true"]},
            ],
        },
    )
    r = load_recipe(p)
    rc = run_recipe(r, only="good")
    assert rc == 0


def test_run_recipe_from_step_skips_earlier(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "bad", "cmd": ["false"]},
                {"name": "good", "cmd": ["true"]},
            ],
        },
    )
    r = load_recipe(p)
    rc = run_recipe(r, from_step="good")
    assert rc == 0


def test_run_recipe_returns_one_on_cmd_failure(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": ["false"]}]})
    r = load_recipe(p)
    assert run_recipe(r) == 1


def test_run_recipe_returns_one_on_python_failure(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "python": "raise ValueError('x')"}]})
    r = load_recipe(p)
    assert run_recipe(r) == 1


def test_run_recipe_only_unknown_step(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": ["true"]}]})
    r = load_recipe(p)
    with pytest.raises(ValueError, match="no step named"):
        run_recipe(r, only="nope")


def test_paper_v1_recipe_loads() -> None:
    """The committed recipe must parse cleanly."""
    repo_root = Path(__file__).resolve().parents[1]
    recipe_path = repo_root / "configs" / "recipes" / "paper_v1.yaml"
    if not recipe_path.exists():
        pytest.skip("paper_v1.yaml not present")
    r = load_recipe(recipe_path)
    assert r.name == "paper_v1"
    assert len(r.steps) >= 5
    names = [s.name for s in r.steps]
    assert "compare" in names
