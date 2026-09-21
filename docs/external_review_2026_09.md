# External review — verified findings (2026-09)

An external reviewer was given a repomix pack of the repo (229 files, ~187k
tokens) with a reviewer's guide framing the run_length=5 sustained-label
result. This document records every claim they made, our verification against
the actual code, and a verdict. It is the artifact to hand to the next
reviewer.

## Method

All claims were checked by grepping the current source and running
diagnostic cells against the real data. No claim was accepted on the
reviewer's word alone. [CONFIRMED], [REFUTED], [PARTIAL] markers below
reflect that verification.

## The result under review

On the sustained run_length=5 label (249 windows, 194 pos / 55 neg,
majority 0.779):

|           | acc@0.5 | calibrated(acc-obj) | AUC   |
|-----------|--------:|--------------------:|------:|
| logreg    | 0.586   | 0.732               | 0.695 |
| rf        | 0.745   | 0.732               | 0.782 |
| rf (5-f)  | 0.722   | 0.714               | 0.738 |

Single-feature ablation showed the RF adds negative value over the
trivial cue: ea_nonzero_frac alone reaches AUC 0.803 (col 23 of the
42-dim tabular vector). See "The dominant artifact" below.

## Confirmed findings

### 1. Fold-boundary leakage [CONFIRMED]

blocked_folds in scripts/compare_baselines.py does not purge or gap the
boundary between folds. With window=5, stride=2, adjacent windows share
3 of 5 frames. Reproducing the repo splitter on a 249-window contiguous
layout:

| num_folds | leaked validation windows |
|----------:|--------------------------:|
| 5         | 16 / 249 (6.4%)           |
| 10        | 36 / 249 (14.5%)          |

test_data_leakage.py exists but tolerates boundary leakage. compare_baselines.py
never calls _leakage_count, so the run JSON does not report it. The 5 to 10
fold AUC increase (0.738 to 0.782) cannot be attributed to "more training
data" alone without a purge gap.

Fix: add a purge gap of at least window_size frames around each fold
boundary, and report leaked_validation_windows in comparison.json.

### 2. Inner calibration split is not blocked [CONFIRMED]

evaluate_model at scripts/compare_baselines.py around line 310:

    rng = np.random.default_rng(seed + k)
    perm = rng.permutation(n_train)
    n_inner = max(int(0.7 * n_train), 1)
    inner_train = train_idx[perm[:n_inner]]
    inner_calib = train_idx[perm[n_inner:]]

The inner split is a random permutation of train indices, not a blocked
split. With window 5 / stride 2, the reviewer measured 59-60 of 60
inner-calibration windows share frames with inner-train in every fold.
The threshold is tuned on near-in-sample scores and does not transfer
to a truly held-out block.

Fix: inner_calib must be a blocked tail (or head) of train_idx, with a
purge gap of window_size frames.

### 3. train_tabular_fold returns y_train as y_val [CONFIRMED]

scripts/compare_baselines.py:118:

    return np.asarray(scores, dtype=np.float64), np.asarray(y_train)  # placeholder

The function is dead code (not called anywhere in the harness), but the
return signature is (val_scores, val_labels) and it returns training
labels. A future refactor that starts calling this would silently
evaluate on the wrong labels.

Fix: delete the function, or return the correct labels.

### 4. comparison.json setup omits config keys [CONFIRMED]

scripts/compare_baselines.py around line 600 builds setup with 14 keys.
The following arguments that affect the result are not recorded:

- label_strategy
- run_length
- fraction_threshold
- calibrate_threshold
- calibration_objective

The 249/194/55 sustained result cannot be reproduced from its own output
JSON without re-reading the command line.

Fix: add all CLI options that can change the label set or evaluation to
the setup block.

### 5. Docs claim is inconsistent with the harness [CONFIRMED]

docs/paper/label_strategies_vntraffic.md:45-46:

> ... standard error ~ 0.042, so the effect is ~5.7 SEs above chance.

The harness itself prints:

> With n=5 folds the minimum achievable Wilcoxon p vs AUC=0.5 is ~0.0625 ...

These contradict. SE-based reasoning assumes independence across folds,
which boundary-leaked folds violate. The "5.7 SEs" claim must be
retracted or replaced with a block-bootstrap CI.

### 6. fold_majority = max(p, 1-p) [CONFIRMED]

scripts/compare_baselines.py:351:

    fold_maj.append(float(max(y_val.mean(), 1 - y_val.mean())))

This is the oracle majority accuracy on the val fold, not the accuracy
of the actual majority-class model (which would predict the training
fold majority on the val fold). When prevalence flips across folds,
these differ. The current "majority baseline" is optimistic.

Fix: train a majority-class predictor on the train fold, predict on
val, compute accuracy.

### 7. > vs >= inconsistency [CONFIRMED]

- Headline accuracy uses scores > 0.5 (line 347).
- _choose_threshold uses scores >= t (line 219).
- Calibrated accuracy uses scores >= threshold (line 355).

Ties on exactly 0.5 are counted differently between the headline and
calibrated numbers.

Fix: pick one and apply consistently.

### 8. --calibrate-threshold help text [CONFIRMED]

scripts/compare_baselines.py:436 says the threshold is chosen "by
Youden's J." The default --calibration-objective is accuracy (since
PR #36). Help text is stale.

### 9. sustained does not require the same pair [CONFIRMED]

scripts/label_strategies.py:171 label_sustained counts consecutive
frames where any pair has a critical TTC. Different pairs on
consecutive frames qualify. The label is more accurately "run of frames
where some conflict exists" than "one persisting conflict."

Fix (optional): add a label_persistent_pair strategy requiring the same
(track_a, track_b) pair across run_length frames.

### 10. The dominant artifact [CONFIRMED — not raised by reviewer]

The reviewer was right that features and label are coupled, but the
mechanism they named (TTC being a feature column) is wrong. Diagnostic:

- ea_min (min over the flattened edge-attr tensor, which contains the
  ttc column) has per-fold AUC 0.517 — chance.
- num_nodes has AUC 0.524 — chance.
- ea_nonzero_frac (col 23 of the tabular vector) has AUC 0.803 —
  above the RF 0.776.

ea_nonzero_frac is the fraction of nonzero cells across the window
[5*num_edges, 15] edge-feature matrix. It correlates at |rho| ~ 0.95
with the zero-fraction of exactly four columns:

| column | rho vs ea_nonzero_frac |
|--------|----------------------:|
| col 3 dvx | -0.952 |
| col 4 dvy | -0.943 |
| col 5 relative_speed | -0.976 |
| col 6 closing_speed | -0.976 |

i.e. ea_nonzero_frac is a proxy for "fraction of edges with any
relative motion." A scene with more relative motion has more
TTC-critical pairs, and therefore more positive labels. Not kinematics.
Not time (rho vs window index = -0.03). Not crowding (rho vs num_nodes
= +0.06).

RF on all 42 features: 0.776. RF on the 3 leaky features only: 0.730.
RF on the 39 clean features: 0.657. The model underperforms a
single scalar.

### 11. rel_heading_sin is always 0 [CONFIRMED — new]

Across one sampled window, rel_heading_sin (edge feature col 9) has
zero_frac = 1.0. Either all actors are modeled with heading 0, or the
field is never populated. Dead feature. The extractor still allocates
a column for it and the aggregate stats over edge_attr are diluted by
it.

Fix: investigate whether heading_rad is populated upstream; if not,
either populate it or remove the column.

## Refuted findings

- "The label uses a different TTC formula from the tested one."
  The _frame_ttc_stats formula (closing_rate / rel_speed_sq) is
  identical algebra to src/graf/ssm/ttc.py:85. Duplicated inline, but
  not different. The feature column ttc uses a different formula
  (compute_ttc_simple, collision-radius quadratic), but that column
  is at chance (see #10).

- "frame_id / track_id / node_frame_index are time proxies that leak."
  In the actual packed data, cols 32-37 and 39 are constant across all
  windows (the extractor writes 0 when the attr is empty), and
  node_frame_idx_mean (col 38) is at AUC 0.476 — below chance. No time
  leak.

- "label_source is silently ignored." It is honoured in build_labels
  (scripts/evaluate_vntraffic.py:189). For non-any label strategies,
  apply_strategy recomputes labels from raw tracks regardless, so the
  flag is inert in that path only.

## Test suite state (at time of audit)

| check | result |
|---|---|
| Unit tests (excl. property) | 516 passed in ~90 s |
| Property tests (Hypothesis) | 43 passed, 1 skipped |
| pre-commit run --all-files | 13/13 hooks pass |
| Coverage (--cov=scripts --cov=src/graf) | 66% overall |
| scripts/compare_baselines.py | covered — 19 unit tests, but no invariants |
| scripts/label_strategies.py | covered for examples |
| src/graf/graph/edges.py | 91% |
| src/graf/models/baselines.py | 94% |
| src/graf/data/graph_dataset.py | 100% |
| train_tabular_fold | NOT COVERED (dead) |
| _leakage_count | covered for evaluate, not for compare_baselines |

Conclusion: all modules are exercised; none of the four blockers
(leakage, inner split, activity scalar, dead return) would have been
caught by the existing tests. The suite tests shape and example
correctness, not pipeline invariants.

## Recommended fix order

1. Purge gap + _leakage_count reporting in blocked_folds and the inner
   calibration split. High priority — invalidates the 5-vs-10 fold
   comparison.
2. Identify and remove ea_nonzero_frac from the extractor, or report
   it as a baseline. The paper cannot claim model signal until the
   trivial cue is separated from the model.
3. Record full setup in comparison.json — five keys, five lines.
4. Fix fold_majority to use the trained majority predictor.
5. Consistent threshold comparison (> vs >=).
6. Retract the "5.7 SEs" claim in the paper doc; replace with a
   block-bootstrap CI or drop.
7. Delete or fix train_tabular_fold.
8. rel_heading_sin: investigate and either populate or remove.

Items 1 and 2 change the result. Items 3-8 are harness and docs hygiene.


## Resolution log

Tracks each confirmed finding against the PR that addresses it. Reviewed
2026-09; last updated after PR #39 and PR #40 merged to `main`.

| # | finding | status |
|---|---------|--------|
| 1 | Fold-boundary leakage in `blocked_folds` | **fixed** in [PR #39](https://github.com/Sharqascc/graf/pull/39) (`79a937e`) — `purge_train_indices` + `leakage_report` helpers; `evaluate_model` purges both gcn and tabular train folds; `comparison.json` records `leakage_raw` and `leakage_after_purge` |
| 2 | Inner calibration split is not blocked | **fixed** in PR #39 — inner split ordered by frame position (70/30) and purged with same gap |
| 3 | `train_tabular_fold` returns `y_train` as `y_val` | **open** — pinned as xfail in `tests/test_pipeline_invariants.py` |
| 4 | `comparison.json` setup omits config keys | **fixed** in PR #39 — `label_strategy`, `run_length`, `fraction_threshold`, `calibrate_threshold`, `calibration_objective`, `purge_gap_frames` now recorded |
| 5 | "5.7 SEs above chance" claim contradicts the harness | **retracted** in the docs PR that added this log |
| 6 | `fold_majority = max(p, 1-p)` is oracle, not trained majority | **open** |
| 7 | `>` vs `>=` inconsistency at 0.5 | **open** |
| 8 | `--calibrate-threshold` help says "by Youden's J" | **open** |
| 9 | `sustained` does not require same (track_a, track_b) pair | **open** (optional) |
| 10 | `ea_nonzero_frac` alone reaches AUC 0.803 | **reproducible** via [PR #40](https://github.com/Sharqascc/graf/pull/40) (`73aa855`) — `--models single_feature --single-feature-name edge_attr_nonzero_frac`; RF without it: `--exclude-features edge_attr_nonzero_frac` |
| 11 | `rel_heading_sin` (edge col 9) always 0 | **open** |

### Effect on reported numbers

The pre-purge RF AUC (0.782) and the "5.7 SEs" claim both predate PR #39.
Finding #1 stands and is now closed in code; the *reported* number is kept
in the paper with an explicit caveat (see README "Known issues and caveats")
until input data is regenerated and the 10-fold purged run produces the
honest number.
