# Research plan

**Status:** active. **Current focus: item 12.**

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
5. **Two tracks.** Science items (1-8, 12-13) change what the paper
   can claim. Coverage items (9-11, 14) raise trust without changing
   any number. Each track is one-at-a-time. Tracks may run in
   parallel because coverage work does not interact with science.

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

status: done
evidence: docs/audit/positive_control.md

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

## Coverage and hygiene

Items 9 through 11 raise the trust floor without changing any number
in the paper. They may be interleaved with items 1-8 during waiting
periods because they do not interact with the science track. Each
still requires an evidence path before being marked done.

### 9. Property tests for `training/conflict_pairs.py`

status: todo
evidence:

Coverage is 44 percent, the lowest in the library, on the file every
downstream result depends on. The functions have clean invariants:
`filter_tracks` never increases row count; `add_world_coords`
preserves (frame_idx, track_id) ordering; CV folds are disjoint and
cover the whole index set; the class-balance helper preserves total
sample count when it reweights.

Done when `docs/audit/conflict_pairs_coverage.md` records the
invariants tested and the new coverage number.

### 10. Smoke tests for the four 0-percent scripts

status: todo
evidence:

`scripts/extract_frames.py`, `scripts/fetch_vntraffic.py`,
`scripts/filter_tracks.py`, `scripts/evaluate_detection.py` sit at 0
percent coverage. Each is invoked as a subprocess in CI, so a silent
regression (missing import, renamed argparse flag, wrong output path)
goes undetected.

Done when `tests/test_scripts_smoke.py` exists and each of the four
scripts appears in at least one test.

### 11. Fix the Docker divergence

status: todo
evidence:

`8 · Docker build + test` failed on four of the last five code PRs
(#131, #132, #133, #134). It is not a required check, so merges are
not blocked, but four in a row is no longer a flake. The image's
environment has drifted from the CI runner's.

Done when `docs/audit/docker_divergence.md` names the cause and
records either the fix or a documented reason the failure is
expected.

## GCN diagnosis and fix

Items 12 and 13 are part of the science track because they change
what the paper's results section can claim. Item 14 is part of the
coverage track because it raises trust without changing any number.

### 12. GCN failure diagnosis: the `GCNConv` finding

status: todo
evidence:

`src/graf/models/gcn_risk.py:96` uses `GCNConv`. That layer has no
`edge_dim` parameter and structurally cannot accept edge features.
The model reads `data.x` and `data.edge_index` (line 106) and never
reads `data.edge_attr`. The 15 edge attributes -- `distance`,
`relative_speed`, `closing_speed`, `ttc`, and 11 more -- do not reach
the model. The GCN sees only the 19 node features, none of which is
TTC.

Its AUC of 0.590 is the ceiling of what those inputs can produce,
not a finding about graph models. The positive control (item 2)
already confirmed the pipeline can learn the rule perfectly -- a
depth-1 tree on `n_critical` hits 1.0. The bottleneck is the model,
not the pipeline.

`tests/test_gcn_risk.py` has 6 tests, all passing, none asserting
that edge features affect the output. Deleting `edge_attr` from
every graph would leave all 6 green. This is the class of test that
coverage does not catch: line coverage was 80 percent, behavior
coverage of the actual failure mode was 0.

Done when `docs/audit/gcn_diagnosis.md` records the code excerpt,
the positive control result, the behavioral test that would have
caught it (`test_output_depends_on_edge_attr`), and the list of the
7 other GCN behavioral tests currently missing.

### 13. Edge-aware GCN fix and re-run

status: todo
evidence:

Replace `GCNConv` with an edge-aware layer: `GINEConv`,
`TransformerConv`, `NNConv`, or `GATv2Conv`, all of which accept
`edge_dim`. Add `edge_dim=15` to every conv. Change the default
pool from `mean` to `max`: mean dilutes a single critical edge
across ~20 vehicles; max preserves the "some pair is critical"
signal that the label asks about.

Add the 8 behavioral tests item 12 names. Re-run
`scripts/compare_baselines.py`. Two outcomes:

- **New GCN AUC >= 0.757** (matches or beats RF): the paper's claim
  flips. The previous result was a bug, not a finding. Results
  section needs a rewrite.
- **New GCN AUC < 0.757**: the claim survives, but with a different
  mechanism. Not "graph models cannot learn this" but "even with
  edge features, message passing does not beat a single aggregated
  scalar on a near-circular label."

Done when `docs/audit/gcn_fix.md` records the new GCN row in
`comparison.json` and `docs/paper/results.md` reflects the corrected
number.

### 14. Step behavioral verification: measure before fixing

status: todo
evidence:

The eight `graf run` steps (`detect`, `track`, `trajectory`,
`graph`, `window`, `train`, `compare`, `audit`) each have test
files that mention their script names -- but referencing a script
is not testing it. The GCN bug is the proof: 6 tests, 80 percent
coverage, and the model was structurally unable to see its own
label signal.

This item is **one measurement, not eight PRs**. For each step,
classify what its tests actually assert: (a) runs the CLI on a
fixture, (b) asserts on the output, (c) only checks arguments or
imports. The likely outcome is 2-3 real gaps, not eight.

Acceptance criteria per step (from the three-reviewer consensus):

  1. detect     -- valid detections, correct class mapping, bbox,
                   confidence, frame indices
  2. track      -- stable IDs, correct association, sensible on
                   detection disappearance
  3. trajectory -- correct ordering, coords, timestamps, missing-
                   frame handling
  4. graph      -- correct nodes, edges, features, loadable .pt
  5. window     -- correct boundaries, stride, labels, sample
                   counts
  6. train      -- loss decreases, outputs finite, checkpoints
                   loadable
  7. compare    -- metrics calculated correctly from the intended
                   predictions and labels
  8. audit      -- detects known problems, produces meaningful
                   report

Detection specifically: inject a fake detector at the model
boundary. Do not download real YOLO in CI -- it is a COCO model,
not UVH-26, and adds network fragility to every run. Test the
code we own: class mapping, bbox validity, JSONL writing, frame
ordering. The real model is tested in the nightly job.

Done when `docs/audit/step_verification.md` records the per-step
classification and either "all eight pass" or a list of the N
steps that need behavioral tests, with each closed as a sub-PR.
