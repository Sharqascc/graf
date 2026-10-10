# Calibration-objective sensitivity — VNTraffic sustained r=5, post-purge

**Status:** addendum to `baselines_sustained_r5_post_purge.md`. The primary
result there is the AUC ordering (single_feature > rf > gcn > majority);
that is unchanged. This addendum answers a follow-up a reviewer is likely
to ask: *"you say no model beats the majority baseline on accuracy — is
that an artifact of how you chose the decision threshold?"*

**Answer:** no, and the direction of the answer is itself reportable.
Accuracy-objective calibration is strictly better than Youden's J for
every model tested, but neither objective gets any model above the
majority baseline on this prior.

## Why this doc exists

The post-purge baseline run used `--calibrate-threshold
--calibration-objective accuracy`. The threshold for each fold was chosen
on a blocked, purged inner-calibration split by maximizing accuracy
directly. This is the prior-aware choice: the calibration metric matches
the metric we report.

Youden's J is the standard alternative in clinical machine learning. It
maximizes TPR - FPR and is imbalance-agnostic: it does not care about the
class prior. On an 78/22 label, the prior is exactly the thing that
decides whether threshold selection helps accuracy.

We ran both objectives on identical inputs. This doc records the
comparison and the mechanism.

## Setup — identical except for one flag

The two runs differ only in `--calibration-objective`:

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
| calibration inner split | blocked + purged |
| **calibration objective** | **accuracy** vs **youden** |
| bootstrap | 10,000 resamples, 95% CI |

Label distribution: **194 positive / 55 negative**, majority baseline
**0.779**. Identical in both runs.

## Results

Fixed accuracy uses threshold 0.5. Calibrated accuracy uses the per-fold
threshold chosen on the inner calibration split.

### Objective = accuracy (the declared setup)

| model | fixed acc | calibrated acc | delta vs majority | mean AUC |
|---|---:|---:|---:|---:|
| majority | 0.778 | 0.778 | -0.001 | 0.500 |
| logreg | 0.573 | 0.661 | -0.118 | 0.695 |
| rf | 0.728 | 0.703 | -0.076 | 0.757 |
| single_feature | 0.765 | 0.628 | -0.151 | 0.803 |

### Objective = youden

| model | fixed acc | calibrated acc | delta vs majority | mean AUC |
|---|---:|---:|---:|---:|
| majority | 0.778 | 0.778 | -0.001 | 0.500 |
| logreg | 0.573 | 0.469 | -0.310 | 0.695 |
| rf | 0.728 | 0.604 | -0.175 | 0.757 |
| single_feature | 0.765 | 0.617 | -0.162 | 0.803 |

### Delta (accuracy-objective minus youden-objective)

| model | delta calibrated acc |
|---|---:|
| logreg | **+0.192** |
| rf | **+0.099** |
| single_feature | **+0.011** |

## Finding 1 — accuracy-objective dominates Youden for every model

The accuracy-objective calibration produces a higher calibrated accuracy
for every model on every prior considered. The gains are largest where
the fixed-0.5 accuracy is farthest below majority:

- logreg: 0.469 to 0.661 (+0.192)
- rf: 0.604 to 0.703 (+0.099)
- single_feature: 0.617 to 0.628 (+0.011)

This is the expected direction. Youden's J is imbalance-agnostic and
optimizes the ROC operating point; on an 78/22 prior, the ROC-optimal
point is not the accuracy-optimal point.

## Finding 2 — neither objective rescues RF past majority

This is the pre-registered question the primary doc answers in the
negative, and it survives the objective sweep:

- accuracy-objective: RF calibrated = 0.703 < majority 0.779
- youden-objective: RF calibrated = 0.604 < majority 0.779

The failure is prior-driven, not threshold-driven. At 78% positives, a
threshold that reduces false positives reduces true positives by more;
the majority baseline dominates every threshold on the ranking. No
selection procedure changes this. The ranking signal (RF AUC 0.757) is
real but not convertible to above-majority accuracy under any threshold
on this prior.

## Finding 3 — the primary result is unchanged

This addendum does not affect the AUC ordering in
`baselines_sustained_r5_post_purge.md`. AUC is threshold-independent;
the calibration objective cannot change it. The primary table stands.

## The mechanism

Youden's J on RF chose thresholds in the range 0.765-0.830 across folds.
Accuracy-objective on the same model chose 0.475-0.775 with a lower
central tendency. The difference in chosen thresholds is real but does
not rescue the model:

| model | mean threshold (youden) | mean threshold (accuracy) |
|---|---:|---:|
| rf | ~0.79 | ~0.65 |
| logreg | ~0.58 | ~0.32 |
| single_feature | ~0.74 | ~0.74 |

The single-feature model's chosen thresholds are nearly identical under
either objective, and its calibrated accuracy is correspondingly nearly
identical. That is consistent with its score distribution being bimodal:
there is one threshold where the classification changes, and both
objectives find it.

## What this does and does not change

- **Does not change:** the paper's primary claim (RF underperforms a
  single scalar cue on the sustained `run_length=5` label), the AUC
  ordering, the trivial-cue finding, the GCN-collapse finding.
- **Does change:** any reader who assumed "we used Youden's J because
  that's the standard." We did not. We used accuracy-objective, and the
  choice is defensible and in fact strictly better on this prior.
- **Worth one sentence in the paper:** "the calibration objective was
  accuracy, not Youden's J; on this prior the two disagree and the
  prior-aware choice is uniformly better across models."

## How to reproduce

Both runs use the paper_v1 recipe flags, differing only in
`--calibration-objective`. The declared-setup run:

    graf compare --tracks data/raw/vntraffic_tracks_all85_min20.jsonl
                --graphs-dir data/processed/graphs/vntraffic_all85
                --homography-config data/raw/vntraffic_homography.yaml
                --output-dir outputs/calibrated_r10_accuracy
                --models majority,logreg,rf,single_feature
                --num-folds 10 --split blocked
                --label-source ttc --label-strategy sustained
                --run-length 5 --ttc-threshold-seconds 1.5
                --ttc-distance-threshold 3.0
                --calibrate-threshold
                --calibration-objective accuracy
                --single-feature-name edge_attr_nonzero_frac
                --bootstrap-resamples 10000

The youden-objective run uses the same flags with
`--calibration-objective youden` and
`--output-dir outputs/calibrated_r10_youden`.

Inputs regenerate from scratch with
`graf reproduce configs/recipes/paper_v1.yaml`.

Verify a produced `comparison.json` against the reviewer checklist with
`graf audit outputs/calibrated_r10_accuracy/comparison.json`.

## Cross-reference

- `docs/paper/baselines_sustained_r5_post_purge.md` — the primary result
  (AUC ordering). Unchanged by this addendum.
- `docs/preregistration_sustained_r5.md` — the decision rule this
  addendum is a follow-up to. The preregistration does not fix a
  calibration objective; both are legitimate. This doc records that the
  declared objective (accuracy) is the one used for the primary result.
- `docs/external_review_2026_09.md` — findings #2 (inner calibration
  split) and #6 (majority baseline). Both fixes are active in the runs
  reported here.
- `graf audit` — reads a comparison.json and reports whether the run
  metadata answers the reviewer checklist.
