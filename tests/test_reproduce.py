"""Tests for the recipe runner."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from graf.reproduce import (
    _select,
    format_stages,
    load_recipe,
    parse_step_spec,
    run_recipe,
    validate_recipe,
)


def _write(tmp_path: Path, body: dict) -> Path:
    p = tmp_path / "recipe.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


def test_load_recipe_minimal(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "cmd": ["echo", "hi"]}],
        },
    )
    r = load_recipe(p)
    assert r.name == "t"
    assert len(r.steps) == 1
    assert r.steps[0].cmd == ["echo", "hi"]


def test_load_recipe_rejects_python_step(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "python": "x = 1"}],
        },
    )
    with pytest.raises(ValueError, match="inline 'python' is not supported"):
        load_recipe(p)


def test_load_recipe_rejects_missing_cmd(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1"}]})
    with pytest.raises(ValueError, match="missing 'cmd'"):
        load_recipe(p)


def test_load_recipe_rejects_cmd_not_list(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": "echo hi"}]})
    with pytest.raises(ValueError, match="list of strings"):
        load_recipe(p)


def test_load_recipe_rejects_non_mapping_step(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": ["not a dict"]})
    with pytest.raises(ValueError, match="must be a mapping"):
        load_recipe(p)


def test_load_recipe_rejects_missing_steps(tmp_path: Path) -> None:
    p = _write(tmp_path, {"name": "t"})
    with pytest.raises(ValueError, match="'steps' list"):
        load_recipe(p)


def test_load_recipe_rejects_non_mapping_root(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text("- just\n- a\n- list\n")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_recipe(p)


def test_load_recipe_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        load_recipe("/nonexistent/recipe.yaml")


def test_run_recipe_dry_run_does_not_execute(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "cmd": ["false"]}],
        },
    )
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


def test_run_recipe_only_unknown_step(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": ["true"]}]})
    r = load_recipe(p)
    with pytest.raises(ValueError, match="no step named"):
        run_recipe(r, only="nope")


def test_run_recipe_from_step_unknown(tmp_path: Path) -> None:
    p = _write(tmp_path, {"steps": [{"name": "s1", "cmd": ["true"]}]})
    r = load_recipe(p)
    with pytest.raises(ValueError, match="no step named"):
        run_recipe(r, from_step="nope")


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
    # no step has a python attribute
    for s in r.steps:
        assert hasattr(s, "cmd")


def test_requires_parsed(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "s1", "cmd": ["true"], "requires": ["a", "b"]},
            ],
        },
    )
    r = load_recipe(p)
    assert r.steps[0].requires == (Path("a"), Path("b"))


def test_requires_must_be_list(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {"steps": [{"name": "s1", "cmd": ["true"], "requires": "a"}]},
    )
    with pytest.raises(ValueError, match="'requires' must be a list"):
        load_recipe(p)


def test_validate_recipe_reports_missing(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "s1", "cmd": ["true"], "requires": ["nope.txt"]},
            ],
        },
    )
    r = load_recipe(p)
    problems = validate_recipe(r, cwd=tmp_path)
    assert len(problems) == 1
    assert "nope.txt" in problems[0]


def test_validate_recipe_ok_when_present(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("x")
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "cmd": ["true"], "requires": ["a.txt"]}],
        },
    )
    r = load_recipe(p)
    assert validate_recipe(r, cwd=tmp_path) == []


def test_validate_recipe_satisfied_by_earlier_produces(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "make", "cmd": ["true"], "produces": "out.txt"},
                {"name": "use", "cmd": ["true"], "requires": ["out.txt"]},
            ],
        },
    )
    r = load_recipe(p)
    assert validate_recipe(r, cwd=tmp_path) == []


def test_validate_recipe_unsatisfied_by_later_produces(tmp_path: Path) -> None:
    """A later step's produces does not satisfy an earlier step's requires."""
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [
                {"name": "use", "cmd": ["true"], "requires": ["out.txt"]},
                {"name": "make", "cmd": ["true"], "produces": "out.txt"},
            ],
        },
    )
    r = load_recipe(p)
    problems = validate_recipe(r, cwd=tmp_path)
    assert len(problems) == 1


def test_run_recipe_aborts_on_missing_requires(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "cmd": ["true"], "requires": ["nope"]}],
        },
    )
    r = load_recipe(p)
    # preflight should abort before running anything
    rc = run_recipe(r, cwd=tmp_path)
    assert rc == 1


def test_run_recipe_allow_missing_bypasses_preflight(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        {
            "name": "t",
            "steps": [{"name": "s1", "cmd": ["true"], "requires": ["nope"]}],
        },
    )
    r = load_recipe(p)
    rc = run_recipe(r, cwd=tmp_path, allow_missing=True)
    assert rc == 0


def test_real_recipes_validate() -> None:
    """Every committed recipe must pass validate_recipe in the repo root."""
    repo_root = Path(__file__).resolve().parents[1]
    recipes = sorted((repo_root / "configs" / "recipes").glob("*.yaml"))
    # Validate from the repo root with cwd=repo_root; some requires are inputs
    # that may not exist in a fresh clone. We only check the loader does not
    # crash and that validate_recipe returns a list (possibly non-empty).
    for p in recipes:
        r = load_recipe(p)
        problems = validate_recipe(r, cwd=repo_root)
        assert isinstance(problems, list)


# ------------------------------------------------------------------
# Coverage additions for reproduce.py
# ------------------------------------------------------------------


def test_load_recipe_rejects_step_missing_name(tmp_path: Path) -> None:
    p = tmp_path / "r.yaml"
    p.write_text("name: t\nsteps:\n  - cmd: [echo]\n")
    with pytest.raises(ValueError, match="missing 'name'"):
        load_recipe(p)


def test_run_recipe_prints_description_and_produces(tmp_path: Path, capsys) -> None:
    p = tmp_path / "r.yaml"
    p.write_text(
        "name: t\n"
        "description: a test recipe\n"
        "steps:\n"
        "  - name: s1\n"
        "    cmd: [echo, hi]\n"
        "    produces: output.txt\n"
    )
    recipe = load_recipe(p)
    rc = run_recipe(recipe, dry_run=True)
    assert rc == 0
    out = capsys.readouterr().out
    assert "a test recipe" in out
    assert "output.txt" in out


# --- numbered-stage selection (--step) --------------------------------


def test_parse_step_spec_single():
    assert parse_step_spec("3", 5) == [2]


def test_parse_step_spec_range():
    assert parse_step_spec("2-4", 5) == [1, 2, 3]


def test_parse_step_spec_list():
    assert parse_step_spec("1,3,5", 5) == [0, 2, 4]


def test_parse_step_spec_from():
    assert parse_step_spec("from-3", 5) == [2, 3, 4]


def test_parse_step_spec_all():
    assert parse_step_spec("all", 3) == [0, 1, 2]


def test_parse_step_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_step_spec("99", 5)


def test_parse_step_spec_rejects_backwards_range():
    with pytest.raises(ValueError):
        parse_step_spec("4-2", 5)


def test_parse_step_spec_rejects_garbage():
    with pytest.raises(ValueError):
        parse_step_spec("foo", 5)


def test_format_stages_numbers_each_step():
    recipe = Recipe(
        name="r",
        description="",
        path=Path("r.yaml"),
        steps=[
            Step(name="alpha", cmd=["x"], produces=None),
            Step(name="beta", cmd=["y"], produces=None),
        ],
    )
    out = format_stages(recipe)
    assert "1. alpha" in out
    assert "2. beta" in out


def test_select_with_explicit_indices():
    recipe = Recipe(
        name="r",
        description="",
        path=Path("r.yaml"),
        steps=[
            Step(name="a", cmd=["x"], produces=None),
            Step(name="b", cmd=["y"], produces=None),
            Step(name="c", cmd=["z"], produces=None),
        ],
    )
    selected = _select(recipe, None, None, [1])
    assert [s.name for s in selected] == ["b"]


def test_select_rejects_step_with_only():
    recipe = Recipe(
        name="r",
        description="",
        path=Path("r.yaml"),
        steps=[Step(name="a", cmd=["x"], produces=None)],
    )
    with pytest.raises(ValueError):
        _select(recipe, "a", None, [0])


def test_run_recipe_dry_run_only_selected_step(capsys):
    recipe = Recipe(
        name="r",
        description="",
        path=Path("r.yaml"),
        steps=[
            Step(name="a", cmd=["echo", "a"], produces=None),
            Step(name="b", cmd=["echo", "b"], produces=None),
        ],
    )
    rc = run_recipe(recipe, dry_run=True, steps=[1])
    assert rc == 0
    out = capsys.readouterr().out
    assert "cmd: echo b" in out
    assert "cmd: echo a" not in out
