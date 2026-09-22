# Pre-registration — sustained `run_length=5` rerun, post-purge

**Written:** 2026-09-22
**Commit at time of writing:** the commit that adds this file to `main`
**Analysis status:** not yet executed. The post-purge number this memo
commits to a decision rule for does not exist in the repo.

## Why this memo exists

The `sustained run_length=5` result in
`docs/paper/label_strategies_vntraffic.md` was produced before the
cross-validation boundary-leak fix in PR #39. The purged rerun has not
been run — the input data
(`data/interim/vntraffic_tracks_all85_min20.jsonl`) lived on an
ephemeral VM and is gone. When the data is regenerated, the primary
comparison will be run once, as declared here.

This memo fixes the comparison before the rerun is executed.
`git log docs/preregistration_sustained_r5.md` will show a commit hash
that predates the post-purge number.

## Primary hypothesis

**The trivial cue `ea_nonzero_frac` alone matches or beats the random
forest on all 42 features, on the sustained `run_length=5` label.**

The single comparison that decides the paper's central claim:

| quantity | command form |
|---|---|
| RF mean AUC | `--models rf --split blocked --num_folds 10 --label-strategy sustained --run-length 5 --calibrate-threshold` |
| single-feature mean AUC | same, `--models single_feature --single-feature-name edge_attr_nonzero_frac` |

Both runs with the purge applied (the code sets `purge_gap_frames` to
the max window span; see `main()` in `scripts/compare_baselines.py`)
and both with bootstrap 95% CI
(`--bootstrap-resamples 10000`).

## Decision rule

Let `RF` be the RF mean AUC point estimate and `CUE` the single-feature
point estimate, each with a 95% bootstrap CI on the per-fold array.

- **CUE CI overlaps RF CI, and CUE point >= RF point:** the finding
  "trivial cue matches or beats the model" holds. Paper reports both
  numbers with their CIs.
- **RF CI sits strictly above CUE CI:** the model adds value over the
  trivial cue on this task. Paper reports this as the headline and
  revises the discussion.
- **CUE CI sits strictly above RF CI:** the model is *worse* than the
  trivial cue. A stronger form of the current finding.

Any of the three outcomes is a valid paper. The memo prevents us from
re-running with adjusted parameters if the first result is unfavorable.

## Analytic choices already made post-hoc

Stated as limitations, not hidden, because they were chosen after
inspecting the label distribution:

1. **`run_length=5` was chosen because it is the only setting that
   produced a non-degenerate label** on this clip.
   `label_strategies_vntraffic.md` shows r=2 and r=3 give 249/0;
   r=5 gives 194/55. The natural-defense argument ("every frame
   critical in a 5-frame window") is sound but is post-hoc.
2. **The trivial-cue correlation was discovered in the audit, not
   pre-specified.** Finding #10 of `external_review_2026_09.md`
   identified `ea_nonzero_frac` (col 23 of the 42-dim vector). The
   comparison is legitimate but retrospective.
3. **The label derives from the same TTC feature column the model
   sees.** The audit established that column is at chance as a
   classifier (finding #10: `ea_min` AUC 0.517), but the coupling of
   label rule and feature family is a design choice, not a
   pre-specification.

## What this memo commits the analysis not to do

Once the rerun is executed:

1. **No sweeping `run_length`.** r=5 is fixed. If the honest number is
   worse than the pre-purge one, we do not search for the r that makes
   RF look best.
2. **No adding or removing feature columns** to chase a number.
   `single_feature` on `ea_nonzero_frac` and `rf` on all 42 columns
   are the declared comparison. `--exclude-features` may be reported as
   a secondary ablation but the primary table is the two rows above.
3. **No redefining the TTC threshold or distance.** 1.5 s / 3.0 m, as
   in the declared setup. Other values are the sensitivity sweep, not
   this analysis.
4. **No switching from `--split blocked` to `--split random`.** The
   purge fix exists because blocked is the honest split.
5. **No sub-sampling the tracks.** All 81 with >=20 frames. Filtering
   to a "cleaner" subset after seeing the number is not an option.

## Deviations

If the rerun requires a deviation from any of the above (e.g. a
version incompatibility forces a different splitter), the deviation is
recorded as a dated addendum to this file, not applied silently.

## Cross-reference

- `docs/external_review_2026_09.md` — findings #1 (purge), #2 (inner
  calibration), #10 (trivial cue)
- `docs/paper/label_strategies_vntraffic.md` — pre-purge result under
  revision; carries the "5.7 SEs" retraction
- `README.md` "Known issues and caveats" — reader-facing summary
- PR #39, PR #43, PR #46 — code changes the rerun will use
