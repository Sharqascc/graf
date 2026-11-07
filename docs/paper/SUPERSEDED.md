# Superseded: pre-B.1 results

Every label-bearing artifact in this directory was generated under
the TCPA-only definition (pre-B.1) described in
`docs/audit/ttc_formula_decision.md`. That definition fires the
critical label on pairs that approach within `distance_threshold` but
never get closer than `dcpa_threshold` - e.g. the 2.8 m / miss-by-2.716 m
counterexample documented in the audit.

On 2026-11-07 the project adopted B.1 (DCPA-gated TCPA). Every
label-bearing artifact here is stale until regenerated.

## How to regenerate

See `RERUN.md` in this directory:

    python scripts/rerun_paper_results.py \
        --with-pixels-per-meter 100 --check
    python scripts/rerun_paper_results.py \
        --with-pixels-per-meter 100 --run

Then inspect `outputs/rerun_b1/` and copy each result back here.
Commit with message `rerun labels under B.1` and delete this file
in that commit.

## How to know it hasn't happened yet

This file exists. When a commit with message
`rerun labels under B.1` removes it, the re-run has happened.
