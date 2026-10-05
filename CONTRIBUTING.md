# Contributing to GRAF

Thank you for your interest in contributing!

## Development setup

## Verification workflow

**CI is the authority on code correctness.** Do not run the test
suite or linters locally before pushing. Write the change, push it,
and let CI report. The one exception is producing an artifact — a
model run, an experiment, a figure — which runs wherever the compute
is and lands as a separate commit from the code that produced it.

See `docs/development_workflow.md` for the full rule.
