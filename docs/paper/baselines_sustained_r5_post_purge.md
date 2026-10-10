# Baseline comparison — VNTraffic, sustained r=5, post-purge

**Status:** the paper's primary result. Produced on the regenerated
VNTraffic inputs with the boundary-leak fix from PR #39 applied and the
pre-registered decision rule from `docs/preregistration_sustained_r5.md`.

## Setup

| parameter | value |
|---|---|
| clip | VNTraffic (Zenodo 18195750) |
| tracks | 81 of 85 (>= 20 frames) |
| windows | 249 (window_size=5, stride=2) |
| label | sustained, run_length=5 |
| TTC threshold | 1.5 s |
| distance threshold | 3.0 m |
| split | blocked, 10 folds |
| purge gap | 5 frames |
| calibration | inner split, blocked + purged, accuracy objective |
| bootstrap | 10,000 resamples, 95% CI |

Label distribution: **194 positive / 55 negative**, majority baseline **0.779**.
Leakage after purge: **0** in every fold, every model.

## Results

| model | mean AUC | 95% CI | mean accuracy | mean F1 |
|---|---:|---|---:|---:|
| majority | 0.500 | [0.500, 0.500] | 0.778 | 0.870 |
| gcn | 0.590 | (see note) | **0.778** | **0.870** |
| logreg | 0.695 | [0.614, 0.782] | 0.573 | 0.629 |
| rf | 0.757 | [0.660, 0.843] | 0.728 | 0.833 |
| single_feature (`ea_nonzero_frac`) | **0.803** | **[0.736, 0.863]** | 0.765 | 0.858 |

### The GCN result

The GCN's accuracy (0.778) and F1 (0.870) are **identical to the
majority baseline** to three decimal places, and its AUC (0.590 ± 0.190)
is barely above chance. The model did not learn the task; it collapsed
to the majority-class predictor. Its per-fold AUC standard deviation
(0.190) is the highest of any model, consistent with per-fold negatives
as low as 2.

This closes the reviewer objection "your models were too weak; try a
graph network." The graph model is second-worst. Both added capacity
(rf over logreg) and added structure (gcn) reduce AUC on this label.

**AUC ordering: single_feature (0.803) > rf (0.757) > gcn (0.590) >
majority (0.500).**

## Pre-purge vs post-purge

| model | pre-purge AUC | post-purge AUC | delta |
|---|---:|---:|---:|
| rf | 0.782 | 0.757 | −0.025 |
| single_feature | 0.803 | 0.803 | 0 |

The random forest loses 0.025 when the boundary leak is removed. The
single-feature cue is unchanged. The gap between the two widens from
0.021 to 0.046.

## Outcome

The two CIs overlap heavily and the cue's point estimate is higher. The
pre-registered decision rule fires **outcome 1**: the trivial cue matches
or beats the model. Neither the RF nor the cue beats the majority
baseline on accuracy. Two negative findings, both pre-registered.

## Calibration-objective follow-up

The threshold calibration above used `--calibration-objective accuracy`
(the prior-aware choice). A companion run with `--calibration-objective
youden` is recorded in
[`calibration_objective_post_purge.md`](calibration_objective_post_purge.md).

Summary: accuracy-objective dominates Youden's J for every model, but
neither objective gets any model above the majority baseline on this
prior. The primary result (AUC ordering) is unchanged by the choice,
because AUC is threshold-independent.
