# Regenerating paper results under B.1

Every artifact here was generated under TCPA-only (pre-B.1). On
2026-11-07 the project adopted B.1. See
`docs/audit/ttc_formula_decision.md` for the decision and
`SUPERSEDED.md` for the scope of what is stale.

## Why no CI workflow

VNTraffic is not redistributable and not tracked (see
`data/README.md`). GitHub Actions cannot see `data/`, so the re-run
must run on the machine that holds
`data/raw/vntraffic_tracks_*.jsonl` and
`data/processed/graphs/vntraffic_*/`.

## Procedure

`scripts/rerun_paper_results.py` reconstructs the exact
`compare_baselines.py` command from each artifact's `setup` block,
and always injects `--ttc-dcpa-threshold 1.5` (the B.1 gate).

Confirm paths resolve first:

    python scripts/rerun_paper_results.py \
        --with-pixels-per-meter 100 --check

Then run:

    python scripts/rerun_paper_results.py \
        --with-pixels-per-meter 100 --run

Outputs land in `outputs/rerun_b1/<artifact_stem>/`. Inspect, then
copy each into `docs/paper/` before committing.

`label_sensitivity_vntraffic.json` is a grid sweep, not a single
comparison. Regenerate it with its own loop over the grid in its
`setup` block. The helper handles the single comparisons.

## After the re-run

1. Commit with message: `rerun labels under B.1`
2. Delete `docs/paper/SUPERSEDED.md` in the same commit
3. Update `docs/paper/results.md` with the new numbers and a B.1 note
4. Re-check margin claims in `docs/paper/conclusion.md`
