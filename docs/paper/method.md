# Method

**Status:** first draft, grounded in the code at commit `7bfb552`. Every
quantity named below is traceable to a function or CLI flag in the repo.
Where the code and the prose disagree, the code is authoritative and the
prose is wrong.

---

## 1. Data

### 1.1 Source

VNTraffic is one of two clips in the Zenodo *Vehicle Tracking* record
(18195750): 1920x1080, 30 fps, 501 frames (~16.7 s), 85 annotated
tracks with frame-level bounding boxes in MOT format, filmed at a
Hanoi intersection with motorbike-heavy mixed traffic. The companion
clip AICC22-Custom is used as an out-of-sample replication (section 6).

### 1.2 Tracks and selection

`scripts/prepare_vntraffic.py` converts the MOT ground truth into
`data/raw/vntraffic_tracks.jsonl`, one row per detection with fields
`{frame_idx, track_id, bbox_xyxy, confidence, actor_class, class_name}`.
The full clip yields 16,036 detection rows across 85 tracks and 501
frames.

Of the 85 tracks, 4 are fragments of fewer than 20 frames (lengths 1,
3, 6, 12 frames). These are excluded, leaving 81 tracks and 16,014
detection rows. The exclusion threshold is fixed at 20 frames before
any model is trained; it is not tuned.

### 1.3 World coordinates

The pipeline needs positions in metres, not pixels. VNTraffic does not
ship a camera calibration, so we use a scale-only homography: the
3x3 matrix `diag(1/PPM, 1/PPM, 1)` with `PPM = 100` pixels per metre,
written to `data/raw/vntraffic_homography.yaml`. The bottom-centre of
each bounding box is projected as the ground-plane anchor point.

**Scale-invariance caveat.** TTC and PET are ratios of lengths to
velocities and are therefore independent of the absolute scale PPM.
DRAC, which has units of m/s^2, is *not* scale-invariant and inherits
the PPM error. This paper reports TTC-based labels only; DRAC is
implemented in `src/graf/ssm/drac.py` but is not part of the reported
result.

## 2. Window construction

The 501-frame clip is cut into overlapping windows of `window_size = 5`
frames with `stride = 2`. This yields 249 windows. Consecutive windows
share 3 of 5 frames.

The overlap is the source of the boundary leak corrected in section 5.
It is retained because it is the field's standard construction; the
correction is applied at split time, not by changing the window rule.

## 3. Features

Each window is summarized by `GraphFeatureExtractor.transform`, which
returns a 42-dimensional vector per window. The columns are named by
`GraphFeatureExtractor.feature_names()`; the vector includes graph
size, node and edge feature statistics, position statistics, and
aggregate label statistics. Column 23, `edge_attr_nonzero_frac`, is
the fraction of nonzero cells in the window's edge-feature matrix.

Every model in the paper sees the same 42-column matrix produced by
this extractor. The single-feature baseline (section 4.4) reads column
23 of the same matrix.

The graph representation itself — nodes for actors, edges for pairs
within a spatial radius — is used by the GCN path. The reported
comparison uses the tabular extractor, because the finding concerns
the label and not the graph structure.

## 4. Models

Four models are reported. All are trained on the same folds, the same
features, and the same label set.

### 4.1 Majority

`MajorityClassBaseline` fits the most common class on the training
fold and predicts it on validation. This is the accuracy floor the
other models must beat.

### 4.2 Logistic regression

`LogisticRegressionBaseline` with `C = 1.0`, `max_iter = 1000`,
`solver = "lbfgs"`, `class_weight = "balanced"`. Used as a linear
reference for whether the signal is linearly separable.

### 4.3 Random forest

`RandomForestBaseline` with scikit-learn defaults. Reads all 42
features. This is the model whose pre-purge AUC of 0.782 was the
paper's original headline.

### 4.4 Single-feature baseline

`SingleFeatureBaseline` scores each window by the raw value of one
named feature column. AUC of this baseline equals AUC of the column
under any monotone transform. It is the trivial-cue comparison: if
this number matches the RF, then the model's apparent signal is
attributable to a single scalar.

The chosen column is `edge_attr_nonzero_frac`. It was identified in
the external review (`docs/external_review_2026_09.md`, finding #10)
as the feature most correlated with the label under a *pre-purge*
analysis. Its selection is retrospective; the pre-registration memo
records this as a limitation (section 7).

## 5. Cross-validation

### 5.1 Blocked folds

The 249 windows are split into 10 contiguous, frame-ordered folds. The
split is produced by `blocked_folds` in `scripts/evaluate_vntraffic.py`.

### 5.2 Purge gap

With `window_size = 5` and `stride = 2`, adjacent windows share 3 of
5 frames. A blocked split that puts adjacent windows on opposite sides
of a fold boundary trains on frames that appear in validation. The
harness corrects this by dropping every training window whose frames
fall within `purge_gap_frames` of any validation window's frame range.
The gap is set to the widest window span; for these settings it is
5 frames. The drop is implemented by `purge_train_indices`.

The uncorrected split leaks 36 of 249 validation windows. After the
purge, the harness reports zero leaked windows in every fold. This is
verified per run in the `leakage_raw` and `leakage_after_purge` fields
of the output JSON.

### 5.3 Inner calibration split

The threshold-calibration procedure fits a throwaway model on a
70/30 inner split of the training fold and selects a decision threshold
on the held-out portion. In earlier versions of the harness this inner
split was a random permutation and leaked adjacent windows exactly as
the outer split did. It is now blocked by frame position and purged
with the same gap. The pre-purge and post-purge difference reported
in section 6 is attributable to both corrections together.

## 6. Results

### 6.1 Primary comparison

The comparison the pre-registration memo commits to is:

> RF on all 42 features vs `SingleFeatureBaseline` on
> `edge_attr_nonzero_frac`, `--split blocked --num_folds 10`,
> sustained `run_length=5` label, purge applied, bootstrap 95% CI
> on each per-fold AUC array.

On VNTraffic, post-purge:

| model | mean AUC | 95% CI | mean accuracy |
|---|---:|---|---:|
| majority | 0.500 | [0.500, 0.500] | 0.778 |
| logreg | 0.695 | [0.614, 0.782] | 0.573 |
| rf | 0.757 | [0.660, 0.843] | 0.728 |
| single_feature | **0.803** | **[0.736, 0.863]** | 0.765 |

Label distribution: 194 positive / 55 negative, majority baseline
0.779. Leakage after purge: 0 in every fold, every model.

### 6.2 Pre-purge vs post-purge

| model | pre-purge AUC | post-purge AUC | delta |
|---|---:|---:|---:|
| rf | 0.782 | 0.757 | -0.025 |
| single_feature | 0.803 | 0.803 | 0 |

The RF AUC drops by 0.025 when the boundary leak is removed. The
single-feature cue is unchanged, because it is a raw feature rather
than a trained model and never benefited from the leak. The gap between
the two widens from 0.021 in the pre-purge table to 0.046 post-purge.

### 6.3 Decision rule outcome

The pre-registration memo defines three possible outcomes for the
RF-vs-cue comparison. With the post-purge numbers, the two CIs overlap
heavily (RF [0.660, 0.843], cue [0.736, 0.863]) and the cue point
estimate is higher. This fires **outcome 1**: the trivial cue matches
or beats the model. The correct statistical statement is that the
trained model does not demonstrably outperform the single scalar.

### 6.4 Majority baseline

Neither the RF nor the cue beats the majority baseline on accuracy
(0.728 and 0.765 against a 0.779 majority). The paper therefore reports
two findings: the model does not beat the cue on AUC, and neither beats
majority on accuracy. Both are negative and both are pre-registered
outcomes of the analysis plan.

### 6.5 Second dataset

[Pending AICC22-Custom run. This subsection is a placeholder until the
second clip has been processed through the same pipeline. If the
finding replicates, it strengthens the paper materially; if it does
not, the paper's scope narrows to VNTraffic and the limitation section
is adjusted accordingly.]

## 7. Limitations

**Sample size.** 249 windows, 10 folds, 2 to 10 negatives per fold.
Bootstrap CIs on the per-fold arrays are correspondingly wide. The
Wilcoxon p against AUC = 0.5 has a minimum achievable value of
2^(1-10) ~ 0.002 at 10 folds; non-significance in any fold is
partially a limit of the fold count.

**Single dataset (until 6.5).** VNTraffic is one 17-second clip. A
negative result on a small single clip is weak evidence about the
underlying task. The AICC22-Custom replication is the mitigation.

**Post-hoc label and cue selection.** The `sustained run_length=5`
label was selected because it is the only setting on this clip that
produces a non-degenerate class split. The `edge_attr_nonzero_frac`
cue was identified in a post-hoc external review. Both are recorded
as limitations in `docs/preregistration_sustained_r5.md`.

**Scale-only homography.** Absolute distance thresholds (DRAC, and any
meter-valued proximity rule) inherit the PPM error. TTC, the reported
indicator, does not.

**Feature extractor is a summary.** `GraphFeatureExtractor` reduces
each window to 42 aggregate statistics and discards the graph
structure. The reported comparison is a comparison of this extractor
against itself; it does not test whether a model reading the graph
directly would perform differently.

## 8. Reproducibility

The full pipeline is three commands from a fresh clone:

```bash
python scripts/fetch_vntraffic.py --data-root data/external/vntraffic
python scripts/prepare_vntraffic.py --dataset-root data/external/vntraffic
python scripts/compare_baselines.py \
    --tracks data/raw/vntraffic_tracks_all85_min20.jsonl \
    --graphs_dir data/processed/graphs/vntraffic_all85 \
    --homography_config data/raw/vntraffic_homography.yaml \
    --output_dir outputs/purged_sustained_r10 \
    --models majority,logreg,rf,single_feature \
    --num_folds 10 --split blocked \
    --label-source ttc --label-strategy sustained --run-length 5 \
    --ttc-threshold-seconds 1.5 --ttc-distance-threshold 3.0 \
    --calibrate-threshold --calibration-objective accuracy \
    --single-feature-name edge_attr_nonzero_frac \
    --bootstrap-resamples 10000
```

The output `comparison.json` records every parameter that affects the
result. `docs/reproducibility.md` documents the full state.

## 9. Cross-reference

- `docs/preregistration_sustained_r5.md` — decision rule, fixed before the rerun
- `docs/external_review_2026_09.md` — audit findings #1, #2, #10
- `docs/mutation_testing_2026_10.md` — test-suite quality diagnostic
- `docs/paper/related_work.md` — literature positioning
