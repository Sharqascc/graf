# Development workflow

**The rule: CI, pre-commit hooks, and the test suite are the authority on
code correctness. Do not run them locally to "verify before pushing."

## The one-way flow

1. Write the change.
2. Commit it. Pre-commit hooks run automatically.
3. Push it.
4. CI runs Ruff, the unit suite, and the property suite on the pushed
   commit.
5. If CI is green, merge.
6. If CI is red, read the failure, fix it, push again.

There is no step where you run `pytest` or `ruff check` in the
development environment before pushing. That duplicates what CI already
does, hides the failure from the record, and wastes time.

## Why

- CI runs the same tools the developer would, on the same files, with
  the same pinned versions. Running locally first adds no information.
- A local pass is not evidence: the local environment drifts from CI's
  (different Python patch version, different transitive packages,
  different pre-commit hook cache). The only pass that means anything
  is the one CI reports.
- The commit log is the audit trail. A change that passed CI has a
  checkmark on GitHub. A change that passed locally has nothing.

## The one exception: producing an artifact

Running a model, an experiment, or a pipeline stage is not a code
correctness check. It is the work itself. That runs wherever the
compute is available — Colab, a cluster, a workstation.

But even then, the order is:

1. Push the code that produces the artifact.
2. Wait for CI to validate it.
3. Run the code to produce the artifact.
4. Commit the artifact (the JSON, the metrics file, the figure).

The code and the artifact land in separate commits, and the code commit
has a CI checkmark before the artifact exists.

## Enforcement

- The `unit-tests` and `property-tests` CI jobs are required for merge.
  Branch protection enforces this where the plan allows; where it does
  not, the reviewer enforces it.
- Pull requests that report local test output as justification for
  skipping CI review are rejected.
- Any commit whose message says "ran pytest locally, all green" is
  rejected — that is not evidence, and it is not the process.

## What this does not mean

- Do not skip pre-commit hooks with `git commit --no-verify`. The hooks
  are the first stage of CI.
- Do not force-push to a branch with a green CI run and expect the run
  to still apply. Force-pushes invalidate the run; CI re-runs on the
  new commit.
- Do not merge a red CI run because the failure "looks unrelated." A
  red run is red; investigate, fix, or revert.

## What "green" means

A commit is green when every check-run on it is in a non-blocking
terminal state:

| conclusion | blocks? | notes |
|---|---|---|
| `success` | no | passed |
| `skipped` | no | conditional job that did not apply (e.g. Dependency review runs on PRs, not on merge commits) |
| `neutral` | no | ran but no verdict |
| `failure` | **yes** | failed |
| `cancelled` | **yes** | cancelled before finishing |
| `timed_out` | **yes** | exceeded the job timeout |
| `action_required` | **yes** | waiting on a human decision |

`skipped` is common and expected. On a merged-main HEAD, the Dependency
review job reports `skipped` because it only runs on pull requests.
Treating it as a failure would block every post-merge run for no reason.

## Enforcement

`scripts/require_ci_green.py` exits 1 if HEAD is not green. Prefix any
cell that runs project code with it:

    python scripts/require_ci_green.py && python <your-script>

It reads a GitHub token from the `GITHUB_TOKEN` environment variable.
