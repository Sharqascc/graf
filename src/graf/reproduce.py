"""Run a recipe: a sequence of subprocess commands.

A recipe is a YAML file with a name and a list of steps. Each step is a
subprocess command (a list of argv tokens) plus optional metadata.

Recipes are deliberately data-only. There is no support for inline
code, embedded shell strings, or Python expressions. Any step that
needs more than a command invocation becomes a script under scripts/.
This keeps the recipe file purely declarative and avoids embedding
executable content in configuration.

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

import yaml


@dataclass
class Step:
    name: str
    cmd: list[str]
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
        if "cmd" not in s:
            raise ValueError(f"step {name!r}: missing 'cmd'")
        if "python" in s:
            raise ValueError(
                f"step {name!r}: inline 'python' is not supported; "
                "write a script under scripts/ and call it via 'cmd'"
            )
        cmd = s["cmd"]
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise ValueError(f"step {name!r}: 'cmd' must be a list of strings")
        produces = Path(s["produces"]) if "produces" in s else None
        steps.append(Step(name=name, cmd=cmd, produces=produces))

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
) -> int:
    """Run the selected steps. Returns 0 on success, 1 on first failure."""
    steps = _select(recipe, only, from_step)
    workdir = cwd or Path.cwd()

    print(f"recipe: {recipe.name}")
    if recipe.description:
        print(f"  {recipe.description}")
    print(f"  {len(steps)} step(s) selected")
    if dry_run:
        print("  (dry run - nothing will execute)")
    print()

    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step.name}")
        if step.produces is not None:
            print(f"    produces: {step.produces}")
        if dry_run:
            print(f"    cmd: {' '.join(step.cmd)}")
            print()
            continue

        result = subprocess.run(step.cmd, cwd=str(workdir))
        if result.returncode != 0:
            print(f"    FAILED: exit {result.returncode}", file=sys.stderr)
            return 1
        print("    ok")
        print()

    print("all steps completed")
    return 0
