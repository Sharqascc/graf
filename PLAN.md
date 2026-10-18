# Research plan

**Status:** active. **Current focus: item 2.**

This file is the single source of truth for what the study is doing next.
It is read by `AGENTS.md` (agent context), by `graf status` output, and by
`tests/test_plan.py` (which fails if a `done` item has no evidence path).

Derived from three reviews: the methodological review (circularity,
density confound, metric choice), the ML-community review (data scarcity,
PR-AUC, density), and the experimental-design review (positive control,
forecasting, label validity). The consensus ranking is in
`docs/audit/consensus_improvements.md`.

---

## Rules

1. **One item at a time.** Do not start item N+1 until item N has a
   `status: done` line and an `evidence:` line naming a file that exists.
2. **Evidence is a path, not a claim.** "It works" is not evidence.
   `docs/audit/positive_control.md` is.
3. **No thresholds swept to chase a number.** If a fix is disappointing,
   record it and move on.
4. **Every change goes through a PR and CI.** No local-only runs.

---

## Items

### 1. Feature audit: is TTC in the input vector?

status: done
evidence: docs/audit/feature_label_circularity.md

The label is "some frame in the window has TTC < 1.5 s". If the 42-dim
feature vector contains TTC, relative speed, closing speed, or distance
computed from the same frames, the model is recovering a thresholded
function of its own inputs. Every review names this as the biggest risk.

Done when `docs/audit/feature_label_circularity.md` names every feature,
marks which are label-derived, and states the verdict.

### 2. Positive control: can any model recover the rule?

status: todo
evidence:

Train a decision tree on `min-TTC` alone. It should reach ~1.0 accuracy.
If it does not, the pipeline cannot learn the trivial rule, and the GCN
failure is a bug — not a finding. **This item gates items 3 through 8.**

Done when `docs/audit/positive_control.md` records the tree's accuracy and
AUC, and states pass/fail.

### 3. Density confound: is the scalar just vehicle count?

status: todo
evidence:

`edge_attr_nonzero_frac` may be a proxy for congestion. Compare it against
plain vehicle count, edge count, and graph density as predictors of the
same label. If they are equivalent, the scalar is a density histogram.

Done when `docs/audit/density_confound.md` has the comparison table.

### 4. Metric suite: PR-AUC, balanced accuracy, calibration

status: todo
evidence:

At 78% positives, accuracy against majority is the wrong headline.
Every model, including the GCN, reports PR-AUC, balanced accuracy, and a
calibration curve alongside ROC-AUC.

Done when `comparison.json` carries the new fields and `graf audit` checks
for them.

### 5. Reword the claim to what the evidence supports

status: todo
evidence:

Replace "graph models cannot learn this" with the scoped claim. A test
asserts the old strong sentence no longer appears in `conclusion.md` or
`results.md`.

### 6. Forecasting variant: features at t, label at t+k

status: todo
evidence:

Detection is near-tautological. The question the introduction actually
asks is whether danger can be flagged *before* it happens. Features from
window t, label from window t+k, TTC-derived features excluded.

### 7. Label sensitivity grid

status: todo
evidence:

Report threshold x window-length as a grid, not one chosen label.
Regenerate `label_sensitivity_vntraffic.md`.

### 8. Language correction: not "pre-registered"

status: todo
evidence:

r=5 was chosen because it was the only setting with a non-empty negative
class. That is choosing by outcome. Add a dated addendum calling it "the
only non-degenerate option", and mark the single-feature baseline
exploratory if it was selected after looking at the 42 features.
