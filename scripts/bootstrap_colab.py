"""One-shot session bootstrap for Colab.

Run once at the start of every Colab session:

    !git clone https://github.com/Sharqascc/graf.git /content/graf 2>/dev/null || true
    !python /content/graf/scripts/bootstrap_colab.py

The script is idempotent: safe to run repeatedly, no-op for steps
that are already done. It exists because a fresh Colab VM lacks
five things this project needs, and every one of them has caused a
wasted debugging cycle at some point:

    1. git user identity  (commits fail with "who are you")
    2. Google Drive mount (data backup unreachable)
    3. the repo at /content/graf  (nothing runs)
    4. the runtime deps: torch, torch-geometric, ultralytics, ruff
    5. GH_TOKEN in Colab secrets (API calls fail)

Each is a 10-second fix. Collectively they were 30+ minutes of
debugging across sessions. This script does all five, prints what
it did, and exits 0 if the environment is ready.

Exit codes
    0  environment ready
    1  environment unusable (missing GH_TOKEN or uninstalled dep
       that could not be installed)
"""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

# ----------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------

def _print_header() -> None:
    print("=" * 70)
    print("GRAF Colab bootstrap")
    print("=" * 70)


def _section(name: str) -> None:
    print(f"\n[{name}]")


def _run(cmd: list[str] | str, cwd: Path | str | None = None) -> int:
    if isinstance(cmd, list):
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    else:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None)
    return r.returncode


def _capture(cmd: list[str] | str, cwd: Path | str | None = None) -> str:
    if isinstance(cmd, list):
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True)
    else:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True)
    return (r.stdout + r.stderr).strip()


def _quiet(cmd: list[str] | str, cwd: Path | str | None = None) -> int:
    if isinstance(cmd, list):
        r = subprocess.run(cmd, cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True)
    else:
        r = subprocess.run(cmd, shell=True, cwd=str(cwd) if cwd else None,
                           capture_output=True, text=True)
    return r.returncode


# ----------------------------------------------------------------
# Step 1 — git identity
# ----------------------------------------------------------------

def step_git_identity(repo: Path) -> None:
    _section("git identity")
    email = _capture(["git", "config", "user.email"], cwd=repo)
    name = _capture(["git", "config", "user.name"], cwd=repo)
    if email and name:
        print(f"  set: {name} <{email}>")
        return
    _run(["git", "config", "user.email",
          "graf-bot@users.noreply.github.com"], cwd=repo)
    _run(["git", "config", "user.name", "graf-bot"], cwd=repo)
    print("  set: graf-bot <graf-bot@users.noreply.github.com>")


# ----------------------------------------------------------------
# Step 2 — Drive mount
# ----------------------------------------------------------------

def step_drive() -> None:
    _section("google drive")
    if Path("/content/drive/MyDrive").exists():
        print("  already mounted")
        return
    if find_spec("google.colab") is None:
        print("  not running in Colab; skipping")
        return
    try:
        from google.colab import drive  # type: ignore[import-not-found]
        drive.mount("/content/drive")
        print("  mounted")
    except Exception as e:
        print(f"  mount failed: {e}")
        print("  (driver-backed data will not be available this session)")


# ----------------------------------------------------------------
# Step 3 — repo presence
# ----------------------------------------------------------------

def step_repo(repo: Path) -> bool:
    _section("repository")
    if not repo.exists():
        print(f"  cloning into {repo}")
        r = _quiet(["git", "clone", "https://github.com/Sharqascc/graf.git",
                    str(repo)], cwd="/content")
        if r != 0:
            print(f"  clone failed (rc={r})")
            return False
    else:
        print(f"  present at {repo}")

    # Sync to main
    _quiet(["git", "fetch", "origin", "--prune"], cwd=repo)
    _run(["git", "checkout", "main"], cwd=repo)
    _run(["git", "reset", "--hard", "origin/main"], cwd=repo)
    head = _capture(["git", "log", "--oneline", "-1"], cwd=repo)
    print(f"  HEAD: {head}")
    return True


# ----------------------------------------------------------------
# Step 4 — dependencies
# ----------------------------------------------------------------

REQUIRED_MODULES = {
    "numpy": "numpy",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "torch_geometric": "torch-geometric",
    "ultralytics": "ultralytics",
    "ruff": "ruff",
}

def step_deps(repo: Path) -> bool:
    _section("dependencies")
    missing = [mod for mod in REQUIRED_MODULES
               if find_spec(mod) is None]
    if not missing:
        print("  all present:", ", ".join(REQUIRED_MODULES.values()))
        return True

    print(f"  installing project + missing: {missing}")
    # Editable install pulls [dev] extras (pytest, hypothesis, ruff,
    # mypy, etc.). torch and torch-geometric are also installed via
    # the project's gpu extra, but for a CPU-only Colab session we
    # want the CPU wheels, so install them explicitly.
    rc = _run([sys.executable, "-m", "pip", "install", "--quiet", "-e",
               f"{repo}[dev]"])
    if rc != 0:
        print("  editable install failed")
        return False

    # CPU-only torch, torch-geometric, ultralytics
    _run([sys.executable, "-m", "pip", "install", "--quiet",
          "torch", "torchvision", "torchaudio",
          "--index-url", "https://download.pytorch.org/whl/cpu"])
    _run([sys.executable, "-m", "pip", "install", "--quiet",
          "torch-geometric", "ultralytics"])

    still = [mod for mod in REQUIRED_MODULES if find_spec(mod) is None]
    if still:
        print(f"  still missing after install: {still}")
        return False
    print("  install complete")
    return True


# ----------------------------------------------------------------
# Step 5 — GH_TOKEN
# ----------------------------------------------------------------

def step_token() -> bool:
    _section("GH_TOKEN")
    # 1. environment variable
    if os.environ.get("GH_TOKEN"):
        print("  present in environment")
        return True
    # 2. Colab userdata
    if find_spec("google.colab") is not None:
        try:
            from google.colab import userdata  # type: ignore[import-not-found]
            tok = userdata.get("GH_TOKEN")
            if tok:
                os.environ["GH_TOKEN"] = tok
                print("  loaded from Colab secrets")
                return True
        except Exception:
            pass
    print("  missing")
    print("  In Colab: left sidebar key icon → add secret named GH_TOKEN")
    print("  → enable 'Notebook access'.")
    return False


# ----------------------------------------------------------------
# Summary
# ----------------------------------------------------------------

def step_summary(repo: Path, ok_repo: bool, ok_deps: bool, ok_tok: bool) -> int:
    _section("summary")
    ready = ok_repo and ok_deps
    if ready:
        print(f"  environment ready; repo at {repo}")
        print("  next: python -m graf.cli doctor")
    else:
        print("  NOT ready:")
        if not ok_repo:
            print("    - repository not cloned")
        if not ok_deps:
            print("    - dependencies missing")
    if not ok_tok:
        print("  (GH_TOKEN missing: GitHub API cells will fail)")
    return 0 if ready else 1


# ----------------------------------------------------------------
# main
# ----------------------------------------------------------------

def main(argv=None) -> int:
    _print_header()
    repo = Path("/content/graf")

    step_git_identity(repo)
    step_drive()
    ok_repo = step_repo(repo)
    ok_deps = False
    if ok_repo:
        ok_deps = step_deps(repo)
    ok_tok = step_token()

    return step_summary(repo, ok_repo, ok_deps, ok_tok)


if __name__ == "__main__":
    raise SystemExit(main())
