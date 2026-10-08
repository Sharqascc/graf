"""Environment and data check for the GRAF pipeline.

`graf doctor` runs this module to answer "is my environment ready?".
It checks dependencies, data directories, and detector weights, and
prints a report with the command to fix anything that is missing.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    hint: str = ""


def check_python() -> Check:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 10)
    detail = f"Python {major}.{minor}.{sys.version_info.micro}"
    hint = "" if ok else "install Python 3.10 or newer"
    return Check("Python version", ok, detail, hint)


def check_packages() -> list[Check]:
    """Check that the pipeline's Python dependencies are importable."""
    packages = [
        ("numpy", "numpy"),
        ("yaml", "pyyaml"),
        ("sklearn", "scikit-learn"),
        ("torch", "torch"),
        ("torch_geometric", "torch-geometric"),
        ("ultralytics", "ultralytics"),
    ]
    out: list[Check] = []
    for mod, dist in packages:
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            out.append(Check(f"package: {dist}", True, f"{mod} {version}"))
        except ImportError:
            out.append(
                Check(
                    f"package: {dist}",
                    False,
                    f"{mod} not importable",
                    f"pip install {dist}",
                )
            )
    return out


def check_dirs(root: Path) -> list[Check]:
    """Check that expected data directories exist."""
    dirs = [
        "data",
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "outputs",
        "configs",
        "configs/recipes",
    ]
    out: list[Check] = []
    for rel in dirs:
        p = root / rel
        out.append(
            Check(f"dir: {rel}", p.exists(), "present" if p.exists() else "missing")
        )
    return out


def check_detectors(root: Path) -> Check:
    """Check whether any detector weights are present."""
    weights_dir = root / "data" / "models"
    if not weights_dir.exists():
        return Check(
            "detector weights",
            False,
            "data/models/ missing",
            "graf fetch-detector --model YOLOv11-S",
        )
    files = list(weights_dir.glob("*.pt"))
    if not files:
        return Check(
            "detector weights",
            False,
            "no .pt files in data/models/",
            "graf fetch-detector --model YOLOv11-S",
        )
    return Check(
        "detector weights",
        True,
        f"{len(files)} file(s): " + ", ".join(f.name for f in files[:3]),
    )


def check_recipes(root: Path) -> Check:
    """Check that at least one recipe is present."""
    rdir = root / "configs" / "recipes"
    if not rdir.exists():
        return Check("recipe files", False, "configs/recipes/ missing")
    recipes = list(rdir.glob("*.yaml"))
    if not recipes:
        return Check("recipe files", False, "no .yaml in configs/recipes/")
    return Check(
        "recipe files",
        True,
        f"{len(recipes)} recipe(s): " + ", ".join(r.name for r in recipes),
    )


def run_checks(root: Path | None = None) -> int:
    """Run all doctor checks. Returns 0 if everything passed, 1 otherwise."""
    if root is None:
        root = Path(__file__).resolve().parents[2]

    checks: list[Check] = []
    checks.append(check_python())
    checks.extend(check_packages())
    checks.extend(check_dirs(root))
    checks.append(check_detectors(root))
    checks.append(check_recipes(root))

    n_ok = sum(1 for c in checks if c.ok)
    n_total = len(checks)

    print(f"GRAF environment check  (root: {root})")
    print("=" * 72)
    for c in checks:
        mark = "[ok]  " if c.ok else "[FAIL]"
        print(f"{mark} {c.name:<30} {c.detail}")
        if not c.ok and c.hint:
            print(f"           fix: {c.hint}")
    print("=" * 72)
    print(f"{n_ok}/{n_total} checks passed")
    if n_ok < n_total:
        print("\nRun `graf fetch-detector` if detector weights are missing.")
        return 1
    return 0
