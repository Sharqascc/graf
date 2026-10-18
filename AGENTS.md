# AGENTS.md

**Read `PLAN.md` before changing anything.**

## Current focus

**Item 1** of `PLAN.md`. Do not start item 2 until item 1 has a
`status: done` line and an `evidence:` path that exists.

## Before you finish any task

- `ruff format .` and `ruff check .` pass
- `pytest -q -m "not hypothesis"` passes
- If you completed a plan item: set `status: done` and put a real path
  in `evidence:`
- Do not sweep thresholds or redefine metrics to improve a number

## Commands

- Reproduce the paper: `graf reproduce configs/recipes/paper_v1.yaml`
- Audit a run: `graf audit outputs/<run>/comparison.json`
- Run the pipeline: `graf run all --site <site>`
- List steps: `graf run --list`

## Do not

- Commit anything under `data/`
- Merge with a red required check
- Edit `docs/paper/manuscript.md` by hand — it is generated
