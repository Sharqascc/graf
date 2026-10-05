"""Refuse to run if the current HEAD is not CI-green.

Enforces the workflow rule: project scripts are only run from commits
that have passed CI. A commit is green when every check-run on it is in
a non-blocking terminal state. A commit with a failure, cancellation,
timeout, or action-required check is not green and this script exits 1.

Reads a GitHub token from the GITHUB_TOKEN environment variable.

Exit codes:
    0   HEAD is green; safe to run scripts
    1   HEAD is not green, or checks have not completed
    2   could not reach the GitHub API, or token missing
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request

OWNER_REPO = "Sharqascc/graf"
NON_BLOCKING = {"success", "skipped", "neutral"}


def head_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def fetch_checks(sha: str, token: str) -> list[dict]:
    url = f"https://api.github.com/repos/{OWNER_REPO}/commits/{sha}/check-runs"
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "require-ci-green/1.0",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode()).get("check_runs", [])


def main() -> int:
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        print("GITHUB_TOKEN not set in environment", file=sys.stderr)
        return 2

    sha = head_sha()
    print(f"HEAD: {sha[:12]}")

    try:
        runs = fetch_checks(sha, token)
    except urllib.error.HTTPError as e:
        print(f"GitHub API error: HTTP {e.code}", file=sys.stderr)
        return 2

    if not runs:
        print("no check-runs for HEAD yet; has CI started?", file=sys.stderr)
        return 1

    incomplete = [r for r in runs if r.get("status") != "completed"]
    if incomplete:
        print(f"{len(incomplete)} check(s) still running:", file=sys.stderr)
        for r in incomplete:
            print(f"  {r['name']}: {r.get('status')}", file=sys.stderr)
        return 1

    blocking = [r for r in runs if r.get("conclusion") not in NON_BLOCKING]
    if blocking:
        print("HEAD is not green:", file=sys.stderr)
        for r in blocking:
            print(f"  {r['name']}: {r.get('conclusion')}", file=sys.stderr)
        return 1

    for r in runs:
        print(f"  {r['name']}: {r.get('conclusion')}")
    print("HEAD is green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
