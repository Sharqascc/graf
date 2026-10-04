#!/usr/bin/env bash
# Arm pre-commit hooks after a fresh Colab clone.
#
# Colab wipes .git/hooks/ on every runtime reset, so the pre-commit
# and pre-push hooks silently stop firing unless they are
# reinstalled. Run this once per session, right after `git clone`.
#
# Usage:
#     bash scripts/setup_colab_hooks.sh

set -euo pipefail

echo 'Installing pre-commit...'
python -m pip install -q pre-commit

echo 'Arming commit and push hooks...'
pre-commit install --hook-type pre-commit --hook-type pre-push

echo 'Pre-warming hook environments (first run may take a minute)...'
pre-commit install-hooks

echo 'Hooks armed. Both stages will fire on the next commit and push.'
