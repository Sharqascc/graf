"""Run a recipe: a sequence of subprocess and inline-Python steps.

A recipe is a YAML file with a name and a list of steps. Each step is
either a subprocess command or a block of inline Python. The recipe is
descriptive, not interpretive: what you see in the YAML is exactly what
runs, in order.

Usage from the CLI:

    graf reproduce configs/recipes/paper_v1.yaml
    graf reproduce configs/recipes/paper_v1.yaml --dry-run
    graf reproduce configs/recipes/paper_v1.yaml --only compare
    graf reproduce configs/recipes/paper_v1.yaml --from-step build-graphs
"""
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Step:
    name: str
    kind: str  # "cmd" or "python"
    cmd: list[str] | None
    code: str | None
    produces: Path | None


@dataclass
class Recipe:
    name: str
    description: str
    path: Path
    steps: list[Step]


def load_recipe(path: str | Path) -> Recipe:
    """Parse a recipe YAML file. Raises ValueError on malformed input."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"recipe not found: {p}")
    raw = yaml.safe_load(p.read_text())
    if not isinstance(raw, dict):
        raise ValueError(f"recipe must be a YAML mapping, got {type(raw).__name__}")
    if "steps" not in raw or not isinstance(raw["steps"], list):
        raise ValueError("recipe must have a 'steps' list")

    steps: list[Step] = []
    for i, s in enumerate(raw["steps"]):
        if not isinstance(s, dict):
            raise ValueError(f"step {i} must be a mapping")
        name = s.get("name")
        if not name:
            raise ValueError(f"step {i} missing 'name'")
        has_cmd = "cmd" in s
        has_py = "python" in s
        if has_cmd == has_py:
            raise ValueError(
                f"step {name!r}: must have exactly one of 'cmd' or 'python'"
            )
        produces = Path(s["produces"]) if "produces" in s else None
        if has_cmd:
            cmd = s["cmd"]
            if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
                raise ValueError(f"step {name!r}: 'cmd' must be a list of strings")
            steps.append(Step(name=name, kind="cmd", cmd=cmd, code=None,
                              produces=produces))
        else:
            code = s["python"]
            if not isinstance(code, str):
                raise ValueError(f"step {name!r}: 'python' must be a string")
            steps.append(Step(name=name, kind="python", cmd=None, code=code,
                              produces=produces))

    return Recipe(
        name=str(raw.get("name", p.stem)),
        description=str(raw.get("description", "")),
        path=p,
        steps=steps,
    )


def _select(recipe: Recipe, only: str | None, from_step: str | None) -> list[Step]:
    steps = recipe.steps
    if only:
        steps = [s for s in steps if s.name == only]
        if not steps:
            names = ", ".join(s.name for s in recipe.steps)
            raise ValueError(f"no step named {only!r}; available: {names}")
        return steps
    if from_step:
        idx = None
        for i, s in enumerate(steps):
            if s.name == from_step:
                idx = i
                break
        if idx is None:
            names = ", ".join(s.name for s in recipe.steps)
            raise ValueError(f"no step named {from_step!r}; available: {names}")
        return steps[idx:]
    return steps


def run_recipe(
    recipe: Recipe,
    *,
    dry_run: bool = False,
    only: str | None = None,
    from_step: str | None = None,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> int:
    """Run the selected steps. Returns 0 on success, 1 on first failure."""
    steps = _select(recipe, only, from_step)
    workdir = cwd or Path.cwd()

    print(f"recipe: {recipe.name}")
    if recipe.description:
        print(f"  {recipe.description}")
    print(f"  {len(steps)} step(s) selected")
    if dry_run:
        print("  (dry run — nothing will execute)")
    print()

    for i, step in enumerate(steps, 1):
        prefix = f"[{i}/{len(steps)}] {step.name}"
        print(f"{prefix}  ({step.kind})")
        if step.produces is not None:
            print(f"    produces: {step.produces}")
        if dry_run:
            if step.kind == "cmd":
                print(f"    cmd: {' '.join(step.cmd or [])}")
            else:
                first = (step.code or "").splitlines()
                print(f"    python: {len(first)} lines")
            print()
            continue

        if step.kind == "cmd":
            assert step.cmd is not None
            result = subprocess.run(step.cmd, cwd=str(workdir), env=env)
            if result.returncode != 0:
                print(f"    FAILED: exit {result.returncode}", file=sys.stderr)
                return 1
        else:
            assert step.code is not None
            ns: dict[str, Any] = {"__name__": f"recipe_step_{step.name}"}
            try:
                exec(compile(step.code, f"<recipe:{step.name}>", "exec"), ns)
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}", file=sys.stderr)
                return 1
        print(f"    ok")
        print()

    print("all steps completed")
    return 0
