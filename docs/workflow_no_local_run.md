# Workflow: never run unverified code

**Rule:** AI-generated code is not executed anywhere until GitHub
CI has checked it. Colab is an editor, not a runtime, until CI is green.

## The flow

1. AI tool produces code.
2. Paste into files. **Do not run anything yet.**
3. `git checkout -b fix-xyz`
4. `git add . && git commit`  (pre-commit fires)
5. `git push`  (pre-push fires: 583 tests + coverage floor)
6. GitHub CI runs 7 jobs on the pushed commit.
7. If red: fix and push again. Colab never ran the code.
8. If green: `git pull`, then in Colab:

    export GITHUB_TOKEN=<read-only PAT>
    python scripts/require_green_ci.py && python the_script.py

## Mechanism 1 — CI job `7 · Import smoke`

`scripts/import_smoke.py` walks every `.py` under `src/graf/` and
`scripts/` and imports each one. Catches `SyntaxError`, `ImportError`,
`NameError` at module level, missing packages — the class of error
tests miss when a broken file is not touched by any test.

## Mechanism 2 — Colab guard

`scripts/require_green_ci.py` reads the current HEAD SHA, queries
GitHub for check-runs of that exact commit, and exits 1 unless all
seven expected jobs are `success` or `skipped`. Fails closed if
`GITHUB_TOKEN` is unset.

## Honest limits

- The guard cannot stop you typing `python the_script.py` alone.
  It makes the safe path the short path; discipline is still required.
- `git push --no-verify` skips local hooks. GitHub CI cannot be
  skipped. See `docs/reproducibility.md`.
- Branch protection is unavailable on a private repo on the free
  plan. The merge cell checks all seven jobs before merging; the
  server does not enforce it. Same reference.
