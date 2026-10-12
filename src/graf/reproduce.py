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
    graf reproduce configs/recipes/paper_v1.yaml --allow-missing

Preflight
---------
Before running anything, the runner validates that each selected step's
`requires:` paths either exist on disk or are produced by an earlier
selected step. Missing inputs abort the run with a single message
listing every problem. Pass `--allow-missing` to skip this check.

`graf validate <recipe>` runs the same check without executing.
"""

from __future__ import annotations

import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Step:
    name: str
    cmd: list[str]
    produces: Path | None
    requires: tuple[Path, ...] = ()


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
        if "python" in s:
            raise ValueError(
                f"step {name!r}: inline 'python' is not supported; "
                "write a script under scripts/ and call it via 'cmd'"
            )
        if "cmd" not in s:
            raise ValueError(f"step {name!r}: missing 'cmd'")
        cmd = s["cmd"]
        if not isinstance(cmd, list) or not all(isinstance(x, str) for x in cmd):
            raise ValueError(f"step {name!r}: 'cmd' must be a list of strings")
        produces = Path(s["produces"]) if "produces" in s else None
        requires_raw = s.get("requires", [])
        if not isinstance(requires_raw, list):
            raise ValueError(f"step {name!r}: 'requires' must be a list of paths")
        requires = tuple(Path(p) for p in requires_raw)
        steps.append(Step(name=name, cmd=cmd, produces=produces, requires=requires))

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


def validate_recipe(
    recipe: Recipe,
    *,
    cwd: Path | None = None,
    only: str | None = None,
    from_step: str | None = None,
) -> list[str]:
    """Return a list of human-readable problems; empty list means runnable.

    For each selected step, every path in `requires:` must either exist
    on disk now, or be the `produces:` of an earlier selected step. The
    check is order-sensitive: a step that produces a path satisfies the
    requires of every step after it, but not of any step before it.
    """
    steps = _select(recipe, only, from_step)
    workdir = cwd or Path.cwd()
    problems: list[str] = []
    produced: set[str] = set()
    for step in steps:
        for req in step.requires:
            if (workdir / req).exists():
                continue
            if str(req) in produced:
                continue
            problems.append(f"step '{step.name}': required input missing: {req}")
        if step.produces is not None:
            produced.add(str(step.produces))
    return problems


def run_recipe(
    recipe: Recipe,
    *,
    dry_run: bool = False,
    only: str | None = None,
    from_step: str | None = None,
    allow_missing: bool = False,
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

    # Preflight: check every selected step's declared inputs.
    if not dry_run and not allow_missing:
        problems = validate_recipe(recipe, cwd=workdir, only=only, from_step=from_step)
        if problems:
            print("preflight failed:", file=sys.stderr)
            for p in problems:
                print(f"  {p}", file=sys.stderr)
            print(
                "\nFix the missing inputs, or pass --allow-missing to skip.",
                file=sys.stderr,
            )
            return 1

    t_total = time.time()
    for i, step in enumerate(steps, 1):
        print(f"[{i}/{len(steps)}] {step.name}")
        if step.produces is not None:
            print(f"    produces: {step.produces}")
        if dry_run:
            print(f"    cmd: {' '.join(step.cmd)}")
            print()
            continue

        t_step = time.time()
        result = subprocess.run(step.cmd, cwd=str(workdir))
        elapsed = time.time() - t_step
        if result.returncode != 0:
            print(
                f"    FAILED: exit {result.returncode}  ({elapsed:.1f}s)",
                file=sys.stderr,
            )
            return 1
        print(f"    ok  ({elapsed:.1f}s)")
        print()

    print(f"all steps completed  ({time.time() - t_total:.1f}s total)")
    return 0
