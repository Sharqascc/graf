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

| model | mean AUC | corrected 95% CI vs 0.5 | mean accuracy | mean F1 |
|---|---:|---|---:|---:|
| majority | 0.500 | [0.500, 0.500] | 0.778 | 0.870 |
| GCN | 0.590 | [0.382, 0.798] | 0.778 | 0.870 |
| logistic regression | 0.695 | [0.614, 0.782] | 0.573 | 0.629 |
| random forest | 0.757 | [0.596, 0.918] | 0.728 | 0.833 |
| single-feature cue | **0.803** | **[0.690, 0.915]** | 0.765 | 0.858 |

Intervals are corrected for cross-validation fold dependence (Nadeau &
Bengio 2003); see `statistical_analysis.md`.

AUC on each fold is computed with the Hanley-McNeil average-rank
formula; the reported interval is the 2.5th to 97.5th percentile of
10,000 bootstrap resamples of the per-fold AUC array, not a parametric
interval on the pooled scores. Pooled AUC is deliberately not reported:
each fold is trained on a different subset and its scores live on a
different scale.

The point ordering is cue > random forest > logistic regression > GCN >
majority. The paired difference between the random forest and the
single-feature cue is −0.046 AUC with a corrected 95% CI of
[−0.138, +0.046] ($p=0.29$): no statistically detectable difference
between the 42-feature model and the single column. A gap as large as
0.14 AUC remains compatible with the data. The correct statement is
that no added value was detected, not that the model adds nothing.

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
more often than the base rate. The models do have ranking signal: the
cue and RF beat chance at corrected $p=0.0002$ and $p=0.0056$
respectively. The paradox is that this ranking signal does not convert
to accuracy above the prior at threshold 0.5. Whether the cause is
calibration, label design, or a property of the class imbalance is not
tested here; it is a hypothesis, not a finding.

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
the majority baseline to three decimal places, and its AUC is 0.590
with corrected 95% CI [0.382, 0.798] and corrected $p=0.3542$ versus
chance. The available folds do not establish above-chance ranking for
the GCN. The point estimate is lower than the cue's by 0.213, at a
level that is borderline under the corrected test ($p=0.056$) and
would cross $\alpha=0.05$ only under the anti-conservative naive test
($p=0.011$). The GCN is not detectably different from the random forest
($p=0.21$).

Adding structure therefore did not produce a detectable improvement
where adding capacity also did not. The point ordering under the
corrected split is cue > random forest > GCN > majority, with the
caveat that the sample size does not support precise separation
between the lower three.

## 6. Pre-registered decision rule

The pre-registration memo defines three outcomes for the primary
comparison, evaluated before the rerun:

1. The two CIs overlap and the cue's point estimate is at least as high.
2. The two CIs overlap and the RF's point estimate is higher.
3. The two CIs do not overlap.

With the post-purge numbers as reported at pre-registration time,
RF = [0.660, 0.843] and cue = [0.736, 0.863]. The intervals overlap
across 0.107 of their joint range and the cue's point estimate is
higher. **Outcome 1 fires.** The pre-registered interpretation is that
the trained model does not demonstrably outperform the single scalar;
the paper's headline claim is the null, not a rejection.

**Post-hoc supplement.** The pre-registered rule compares independent
intervals, which is not a valid test of difference. The paired
analysis reported in `statistical_analysis.md` replaces that comparison
with a corrected resampled $t$-test on the per-fold differences. It
reaches the same conclusion: $\bar\Delta$AUC = −0.046, corrected 95%
CI [−0.138, +0.046], $p=0.29$, no detectable difference. The
pre-registered rule is retained here verbatim to preserve the audit
trail; the paired test is the statistically correct statement of what
the data show.

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

**This dataset is reported descriptively only.** One fold has zero
validation negatives, so effective $n=4$. The corrected 95% CI on the
RF − cue paired difference spans [−1.618, +1.220] — 2.8 AUC units on a
metric bounded in [0, 1] — and no inferential claim is made on this
fold count. Per-fold RF − cue differences: −0.944, −0.429, +0.056,
+0.522. Mean ΔAUC: −0.199. The direction is consistent with VNTraffic;
the magnitude is not interpretable at $n=4$.

## 8. Summary of results

1. On VNTraffic, no statistically detectable difference was found
   between the random forest (AUC 0.757) and the single-feature cue
   (AUC 0.803): paired $\Delta$AUC = −0.046, corrected 95% CI
   [−0.138, +0.046], $p=0.29$. A gap as large as 0.14 remains compatible
   with the data.
2. Both the cue and the random forest have statistically significant
   ranking signal against chance ($p=0.0002$ and $p=0.0056$,
   corrected). Neither exceeds the majority-class accuracy at threshold
   0.5. The reason is untested.
3. The GCN's accuracy equals the majority baseline to three decimals
   and its AUC does not establish above-chance ranking at this fold
   count ($p=0.35$). It is not detectably different from the random
   forest.
4. The random forest loses 0.025 AUC when the boundary leak is
   corrected; the cue is unaffected.
5. On the second clip, the direction is consistent with VNTraffic but
   the fold count (effective $n=4$) does not support inference.

Interpretation and implications for the label definition and for the
SSM literature are deferred to `discussion.md`.
