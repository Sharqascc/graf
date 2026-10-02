# Results

**Status:** first draft, grounded in the frozen JSON artifacts under
`docs/paper/baselines_*.json` and the tables in `baselines_*.md`. Every
number below is copied from those files unchanged. If the JSONs and the
prose disagree, the JSONs are authoritative.

---

## 1. Setup

The primary comparison is the one the pre-registration memo commits to
(`docs/preregistration_sustained_r5.md`). It is:

> random forest on all 42 `GraphFeatureExtractor` columns, versus
> `SingleFeatureBaseline` on `edge_attr_nonzero_frac`, under blocked
> 10-fold CV with the purge gap applied, on the sustained
> `run_length=5` label.

Data: VNTraffic (Zenodo record 18195750), 81 tracks, 249 windows,
194 positive / 55 negative, majority baseline 0.779. Boundary leakage
after the purge is zero in every fold and every model; this is verified
per run in the `leakage_after_purge` field of the output JSON.

## 2. Primary comparison

| model | mean AUC | 95% CI | mean accuracy | mean F1 |
|---|---:|---|---:|---:|
| majority | 0.500 | [0.500, 0.500] | 0.778 | 0.870 |
| GCN | 0.590 | (see §5) | 0.778 | 0.870 |
| logistic regression | 0.695 | [0.614, 0.782] | 0.573 | 0.629 |
| random forest | 0.757 | [0.660, 0.843] | 0.728 | 0.833 |
| single-feature cue | **0.803** | **[0.736, 0.863]** | 0.765 | 0.858 |

AUC on each fold is computed with the Hanley-McNeil average-rank
formula; the reported interval is the 2.5th to 97.5th percentile of
10,000 bootstrap resamples of the per-fold AUC array, not a parametric
interval on the pooled scores. Pooled AUC is deliberately not reported:
each fold is trained on a different subset and its scores live on a
different scale.

The point ordering is cue > random forest > logistic regression > GCN >
majority. The trained models do not exceed the single scalar that reads
one column of the feature matrix they themselves are trained on.

## 3. Neither model beats the majority baseline on accuracy

The accuracy column reports the same thing from a different angle.
Majority is 0.778. The random forest reaches 0.728; the cue reaches
0.765. Neither crosses the floor. This is not a thresholding artifact:
the calibrated variants in `baselines_sustained_r5_post_purge.md` §3
also fall below 0.779 under both the Youden objective and the
accuracy objective (numbers in that file, not repeated here).

The label is 78% positive. A constant-positive predictor therefore has
high accuracy by construction. That the RF does not exceed it means the
model's ranking is not strong enough to justify predicting positive
more often than the base rate — not that the model is at chance (its
AUC is 0.757), but that no threshold on its scores yields a
classification rule better than the prior.

## 4. Pre-purge versus post-purge

| model | pre-purge AUC | post-purge AUC | delta |
|---|---:|---:|---:|
| random forest | 0.782 | 0.757 | −0.025 |
| single-feature cue | 0.803 | 0.803 | 0 |

The random forest loses 0.025 when the boundary leak documented in
`docs/external_review_2026_09.md` finding #1 is corrected. The cue does
not move, because it is a raw column rather than a fitted model and
never benefited from the leak. The gap between the two widens from
0.021 pre-purge to 0.046 post-purge.

This delta is the single most important number in the paper. It is the
reason the pre-registration memo was written before the rerun, and it
is the reason the primary comparison in §2 is the corrected one.

## 5. The GCN collapses to the majority class

The GCN path exists to close the reviewer objection that the tabular
models are too weak and a graph network would perform differently.
It does not. The GCN's accuracy (0.778) and F1 (0.870) are identical to
the majority baseline to three decimal places. Its AUC is 0.590 ± 0.190,
barely above chance and with the highest per-fold standard deviation
of any model. The model did not learn the task; it degenerated to the
majority-class predictor.

Adding structure therefore does not help where adding capacity did not.
The complete AUC ordering under the corrected split is cue > random
forest > GCN > majority. Both an increase in model capacity (RF over
logistic regression) and the introduction of graph structure (GCN)
reduce ranking performance on this label.

## 6. Pre-registered decision rule

The pre-registration memo defines three outcomes for the primary
comparison, evaluated before the rerun:

1. The two CIs overlap and the cue's point estimate is at least as high.
2. The two CIs overlap and the RF's point estimate is higher.
3. The two CIs do not overlap.

With the post-purge numbers, RF = [0.660, 0.843] and cue = [0.736,
0.863]. The intervals overlap across 0.107 of their joint range and
the cue's point estimate is higher. **Outcome 1 fires.** The
pre-registered interpretation is that the trained model does not
demonstrably outperform the single scalar; the paper's headline claim
is the null, not a rejection.

## 7. Second dataset

AICC22-Custom (the second clip in the same Zenodo record) was processed
through the same pipeline with the same hyperparameters. Its label
distribution is different — 97 windows, 69 positive / 28 negative,
majority 0.711 — and one fold has zero validation negatives, so the
effective fold count for AUC is 4.

| model | AUC | 95% CI |
|---|---:|---|
| majority | 0.500 | [0.500, 0.500] |
| logistic regression | 0.611 | [0.291, 0.930] |
| random forest | 0.567 | [0.243, 0.870] |
| single-feature cue | **0.765** | [0.534, 0.950] |

Direction replicates: the cue leads the random forest by 0.199, larger
than the 0.046 gap on VNTraffic. The absolute CIs are too wide for a
confirmatory claim — neither model's interval excludes 0.5 — so the
second dataset is reported as directional evidence only. The cue is the
more clip-stable estimator: 0.803 on VNTraffic, 0.765 on AICC22-Custom,
versus the RF's 0.757 and 0.567.

## 8. Summary of results

1. The single-feature cue has the highest AUC of any model on either
   dataset.
2. No model beats the majority baseline on accuracy on either dataset.
3. The random forest loses 0.025 AUC when the boundary leak is
   corrected; the cue is unaffected.
4. A graph network with more capacity and more structure performs
   worse than the tabular baselines, and collapses to the majority
   predictor.
5. The direction of the primary comparison replicates on a second clip,
   but at low power.

Interpretation and implications for the label definition and for the
SSM literature are deferred to `discussion.md`.
