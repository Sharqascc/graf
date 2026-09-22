# Experiment tracking

Experiments are tracked as immutable JSON artifacts under version
control — not in a database or web service. The derivation of any
number in `docs/paper/` is recoverable from `git log`.

## Where results live

| file | what |
|---|---|
| `outputs/<run>/comparison.json` | raw output of one `scripts/compare_baselines.py` invocation |
| `docs/paper/*.json` | JSON backing a paper table |
| `docs/paper/*.md` | human-readable companion to a JSON |
| `docs/real_data_metrics.json` | historical real-data numbers |
| `docs/external_review_2026_09.md` | audit of published claims against code |
| `docs/mutation_testing_2026_10.md` | test-suite quality diagnostic |

A paper table `docs/paper/foo.md` is generated from
`docs/paper/foo.json`, which is copied from an `outputs/<run>/` run
after review. The run's CLI is recorded in the JSON `setup` block, so
the table is reproducible by re-running the recorded command.

## What `setup` records

Every `comparison.json` records the parameters that can change the
result:

```
tracks, graphs_dir, window_size, stride,
label_source, label_strategy, run_length, fraction_threshold,
ttc_threshold_seconds, ttc_distance_threshold,
split, num_folds, purge_gap_frames,
calibrate_threshold, calibration_objective,
epochs, batch_size, lr, pos_weight, seed
```

Audit finding #4 records the point in time when only 14 of these were
present; PR #39 added the label and calibration parameters.

## Reproducing a past number

1. Read the `setup` block from the JSON backing the paper table.
2. Re-run `scripts/compare_baselines.py` with those flags.
3. Compare against the `results` block.

Divergence is either a code change since the run (see `git log` for the
JSON) or a data change. Data provenance is the weak link — see
`docs/reproducibility.md`.

## Pre-registration

Before running an analysis that will be reported, its decision rule is
committed as a memo under `docs/preregistration_*.md`. Example:
`docs/preregistration_sustained_r5.md`. `git log` on that file shows a
commit hash that predates the number it constrains. That is the entire
audit value; a reviewer does not need to trust the authors' word that
the plan existed first.

## Cross-reference

- `docs/paper/` — result tables and their JSON
- `docs/external_review_2026_09.md` — audit of what was tracked and what was not
- `docs/preregistration_sustained_r5.md` — the one pre-registered analysis
- `docs/mutation_testing_2026_10.md` — a diagnostic, not a result
