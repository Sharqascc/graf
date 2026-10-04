"""Refuse to proceed unless HEAD has all seven CI jobs green.

Usage in Colab, before running anything:

    export GITHUB_TOKEN=<read-only PAT>
    python scripts/require_green_ci.py && python the_thing.py

The guard exits 0 if the current commit's CI is fully green, 1
otherwise. If no token is set, the guard fails closed rather than
open: not knowing means not running.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

EXPECTED = {
    "1 · Ruff",
    "2 · Unit tests",
    "3 · Property tests (Hypothesis)",
    "4 · Invariants (pre-push suite)",
    "5 · Manuscript build",
    "6 · Dependency review",
    "7 · Import smoke",
}


def current_sha(repo: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo,
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def current_repo() -> str:
    url = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    if url.startswith("https://github.com/"):
        url = url[len("https://github.com/"):]
    elif url.startswith("git@github.com:"):
        url = url[len("git@github.com:"):]
    if url.endswith(".git"):
        url = url[:-4]
    if "@" in url:
        url = url.split("@", 1)[1]
    return url


def api(path: str, token: str):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "graf-ci-guard",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode() or "{}")


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    sha = current_sha(repo)
    slug = current_repo()
    print(f"checking CI status for {slug} @ {sha[:8]}")

    token = os.environ.get("GITHUB_TOKEN", "").strip()
    if not token:
        print("GITHUB_TOKEN not set — refusing to proceed (fail closed)")
        return 1

    try:
        runs = api(f"/repos/{slug}/commits/{sha}/check-runs", token)
    except urllib.error.HTTPError as e:
        print(f"GitHub API error: {e.code} {e.reason}")
        return 1

    checks = runs.get("check_runs", [])
    got = {c["name"]: c.get("conclusion") for c in checks}
    missing = EXPECTED - set(got)
    failed = [n for n, c in got.items() if c not in ("success", "skipped")]

    print(f"found {len(checks)} check runs:")
    for name in sorted(got):
        print(f"  {name:<32} {got[name]}")

    if missing:
        print(f"\nMISSING expected checks: {sorted(missing)}")
    if failed:
        print(f"\nFAILING checks: {sorted(failed)}")

    if missing or failed:
        print("\nREFUSING TO RUN: CI not fully green at this commit.")
        return 1

    print("\nAll expected checks passed. Safe to proceed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
