"""Import every module under src/ and scripts/.

Catches the class of error that a test suite cannot: syntax errors,
import errors, missing dependencies, and references to modules that
no longer exist. If a new file is added and nothing imports it, no
test will exercise it — but this script will.

Run locally:  PYTHONPATH=src python scripts/import_smoke.py
In CI:        same command, as job "7 · Import smoke".
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))


def collect() -> list[str]:
    """Every importable module name under src/graf and scripts/."""
    import graf

    names: list[str] = [
        mod.name for mod in pkgutil.walk_packages(graf.__path__, prefix="graf.")
    ]

    for p in sorted((REPO / "scripts").glob("*.py")):
        if p.name == "__init__.py":
            continue
        names.append(f"scripts.{p.stem}")
    return names


def main() -> int:
    names = collect()
    failures: list[tuple[str, str]] = []
    for name in names:
        try:
            importlib.import_module(name)
        except Exception:
            failures.append((name, traceback.format_exc().strip()))

    print(f"attempted {len(names)} imports")
    if not failures:
        print("all modules imported cleanly")
        return 0

    print(f"\n{len(failures)} failure(s):\n")
    for name, tb in failures:
        print(f"=== {name} ===")
        print(tb)
        print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
