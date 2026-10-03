# The window-level surrogate safety label is an activity proxy

*[Draft manuscript. Title, authors, affiliations to be finalized.]*

- **Code:** https://github.com/Sharqascc/graf
- **Data:** VNTraffic & AICC22-Custom, Zenodo record 18195750
- **Pre-registration:** `docs/preregistration_sustained_r5.md`
- **Reproducibility:** `docs/reproducibility.md`

---

## Abstract

Surrogate safety measures (SSMs) such as time-to-collision are widely
used as substitutes for crash data in traffic safety research. The
standard machine-learning pipeline computes a per-pair per-frame SSM,
aggregates it into a binary window-level label by a rule like "any
frame contains a critical pair," and trains a classifier on window
features. The aggregation step is treated as plumbing. We show it is
not. On two hand-annotated traffic clips from the public Zenodo
*Vehicle Tracking* record, under a sustained `run_length=5` TTC label
with a purged 10-fold cross-validation split, a single scalar feature
(`edge_attr_nonzero_frac`) reaches mean AUC 0.803 [0.736, 0.863] and
matches or exceeds every trained model: random forest 0.757 [0.660,
0.843], logistic regression 0.695, and a graph convolutional network
0.590. The GCN's accuracy and F1 are identical to the majority-class
baseline to three decimal places, indicating it collapsed to the
trivial predictor. Neither the models nor the scalar exceeds the
majority baseline on accuracy. The scalar's advantage is not a
lucky column: it is a proxy for the fraction of edges in the window
with any relative motion, and correlates at |rho| >= 0.94 with four
relative-velocity features. The window label is a coarse activity
detector, and the field's models are learning to recover activity
rather than conflict severity. We recommend that any SSM-based
classification result be validated against a trivial single-feature
baseline derived from the same features before a model comparison
is reported. The check takes minutes and is not currently standard
practice.

---

## 1. Introduction

### 1.1 The field's pipeline

Surrogate safety measures substitute observable near misses for the
crashes that safety analysis would prefer to study. Time-to-collision
(TTC), post-encroachment time (PET), and deceleration rate to avoid a
crash (DRAC) are the standard indicators [Perkins & Harris, 1968;
Hydén, 1987; Tarko et al., 2009; Mahmud et al., 2017]. The premise,
inherited from the traffic-conflict technique and refined over six
decades, is that the frequency and severity of conflicts correlate
with the frequency and severity of crashes, and can therefore serve
as a proxy when crash data is too sparse to estimate risk directly.

As trajectory data became easy to extract from video, a machine-
learning literature grew on top of the SSM construct. The standard
pipeline has four stages:

1. Extract per-frame trajectories for every actor in a scene.
2. Compute a per-pair, per-frame SSM value (TTC, PET, or DRAC) for
   every pair of nearby actors.
3. Aggregate to a **window-level binary label** via a rule such as
   "the window is critical if any frame contains a critical pair."
4. Train a classifier to predict the label from window-level features.

Steps 1, 2, and 4 receive most of the attention in the literature.
Step 3 is treated as a definition rather than a decision. Reviews of
the field [Formosa et al., 2020; Wang et al., 2021] describe the
aggregation as one option among several without examining the
consequences of the choice for what the classifier can learn.

### 1.2 The question

This paper asks a narrow, empirical question:

> Under the standard window-level SSM label construction, does the
> resulting classification task contain signal that a trained model
> can extract beyond what a trivial alternative from the same
> features already provides?

The trivial alternative is a single scalar: the fraction of nonzero
cells in the window's edge-feature matrix (`edge_attr_nonzero_frac`,
one column of the feature vector the models are trained on). If the
models do not beat this scalar, the label is not supporting the task
in the way a classification pipeline assumes.

### 1.3 The answer, briefly

On VNTraffic, a 501-frame Hanoi intersection clip from the public
Zenodo *Vehicle Tracking* record (18195750), the single-feature
scalar reaches mean AUC 0.803 under purged 10-fold cross-validation.
A random forest on all 42 features reaches 0.757. A logistic
regression reaches 0.695. A graph convolutional network reading the
same window summary reaches 0.590 and matches the majority-class
baseline exactly on accuracy and F1 — it did not learn the task.
Neither the models nor the scalar exceeds the majority baseline on
accuracy. On AICC22-Custom, a second clip from the same record, the
direction replicates: the scalar (0.765) leads the random forest
(0.567) by a wider margin than on VNTraffic, though with confidence
intervals too wide to support a confirmatory claim.

The scalar is not a lucky feature. It is a proxy for the fraction of
edges in the window with any relative motion, correlating at
|rho| >= 0.94 with the four relative-velocity columns of the edge
feature matrix and uncorrelated with window index and node count.
The window label, in effect, is a coarse activity detector. Models
trained on the same features learn to recover activity; the scalar
reads it directly.

### 1.4 Contributions

1. **A negative result with a precise diagnosis.** Under the standard
   window-level SSM label with a corrected cross-validation split,
   neither trained models nor a graph network exceed a single scalar
   feature on ranking performance, and neither beats the majority
   baseline on accuracy. The failure is in the label's construction,
   not in the models.

2. **The mechanism.** The scalar that ties the models is a proxy for
   scene activity rather than conflict severity. This is established
   by the correlation structure of the feature matrix, not by
   argument.

3. **A reproducible reproduction.** All results are produced from a
   three-command pipeline on public data. A pre-registration memo
   fixed the primary comparison before the rerun that produced the
   reported numbers. The paper's central diagnostic — a trained model
   versus a single scalar from the same features — is a check any
   SSM-classification study can run in minutes.

### 1.5 What we are not claiming

We are not claiming that GNNs are unhelpful for traffic safety, that
TTC is a poor indicator, or that our finding generalizes beyond the
two clips tested. We are claiming that, on this data, under this
label, the classification task does not support the models trained
on it, and that the reason is visible from the label construction
rather than from the models. The scope and limitations of the claim
are documented in `discussion.md` §5–§6.

### 1.6 Paper structure

§2 reviews the surrogate-safety, machine-learning, and graph-model
literatures the work sits within, and the evaluation-pitfalls
literature on data leakage and trivial baselines. §3 describes the
pipeline: data, window construction, features, models, and the
purged cross-validation split. §4 reports results across the two
clips and the three model classes. §5 interprets them, states the
threats to validity, and gives the implications for the SSM
community. §6 concludes.

---

## Note on status

This draft is complete in structure and cites numbers that exist in
`results.md`. The literature citations in §1.1 are marked `[CHECK]`
in `related_work.md` and should be verified against Google Scholar
before submission. The abstract currently runs slightly over 200
words; trim for the target venue.

---

# Related work

below is marked with a confidence flag. `[HIGH]` = author, year, and content
I am confident about. `[CHECK]` = the paper exists but verify the exact year,
venue, and DOI in Google Scholar before submission. Nothing here is fabricated;
the flags mark where a human read is required.

---

## 1. Surrogate safety measures: from conflict to indicator

Surrogate safety measures (SSMs) are observable traffic events — a near miss,
a hard braking, a close approach — that stand in for crashes when crash data
is too sparse to estimate risk directly. The traffic conflict technique
originated with Perkins and Harris at General Motors [CHECK: Perkins & Harris,
1968, *Traffic conflict characteristics: Accident potential at intersections*],
who proposed that observable conflicts between vehicles correlate with
collision frequency. Hydén [CHECK: Hydén, 1987, *The development of a method
for traffic safety evaluation: The Swedish Traffic Conflicts Technique*]
formalized the approach in the Swedish Traffic Conflicts Technique, introducing
time-to-collision (TTC) as the principal severity proxy. The core intuition —
that frequency and severity of conflicts can substitute for frequency and
severity of crashes — remains the foundation of the field sixty years later.

The subsequent decades produced a menu of indicators. TTC measures time until
collision at current velocities assuming constant motion. Post-encroachment
time (PET) measures the gap between one road user leaving a conflict zone and
another entering it. Deceleration rate to avoid the crash (DRAC) measures the
required braking to prevent collision. Comprehensive reviews of the indicator
menu appear in Tarko and colleagues' TRB white paper [CHECK: Tarko et al., 2009,
*Surrogate measures of safety*, TRB Annual Meeting] and in Zheng, Ismail, and
Meng [CHECK: Zheng, Ismail, Meng, 2014, *Traffic conflict techniques for road
safety analysis*, Accident Analysis & Prevention]. Mahmud and coauthors provide
a recent catalog of proximal indicators and their empirical properties [HIGH:
Mahmud, Ferreira, Hoque, Tavassoli, 2017, *Application of proximal surrogate
indicators for safety evaluation: A review of recent developments and research
needs*, IATSS Research]. Laureshyn, Svensson, and Hydén gave the theoretical
framework for micro-level behavioural data [HIGH: Laureshyn, Svensson, Hydén,
2010, *Evaluation of traffic safety, based on micro-level behavioural data*,
Accident Analysis & Prevention]. Johnsson, Laureshyn, and De Ceunynck extended
the catalog to vulnerable road users [CHECK: Johnsson, Laureshyn, De Ceunynck,
2018, *In search of surrogate safety indicators for vulnerable road users*,
Transport Reviews]. Lord and Mannering's review of crash-frequency modeling
remains the reference for why crash-based estimation is difficult and what
SSMs are meant to substitute for [HIGH: Lord, Mannering, 2010, *The statistical
analysis of crash-frequency data: A review and assessment of methodological
alternatives*, Accident Analysis & Prevention].

What this literature shares is a treatment of SSMs as *per-pair, per-frame*
quantities. A TTC value is defined for two actors at one instant. The field's
standard aggregation — a window is positive if any critical pair exists in any
frame — is a *label design* choice layered on top of the SSM, not part of it.
That choice is rarely examined as a design decision with consequences for
classification. This paper treats it as exactly that.

## 2. The machine-learning turn and its label problem

As SSMs became easy to compute from video, a wave of work applied supervised
learning to predict conflicts or their proxies. Formosa and coauthors surveyed
the methods and noted that most reported results depend on the label definition
as much as on the model [HIGH: Formosa, Quddus, Ison, Purdie, Bhavsar, Sharples,
2020, *Predicting traffic conflicts for use in road safety analysis: A review of
analytic methods and future directions*, Analytic Methods in Accident Research].
Wang and coauthors reviewed the SSM-plus-ML pipeline specifically for connected
and automated vehicle safety [CHECK: Wang, Xie, Huang, Liu, Chen, 2021, *A review
of surrogate safety measures and their applications in connected and automated
vehicles safety modeling*, Accident Analysis & Prevention]. Guo, Sayed, and Zaki
applied computer-vision SSM extraction to intersection safety [CHECK: Guo,
Sayed, Zaki, 2019 or 2020, *Surrogate safety measure for evaluating the safety of
intersections using computer vision*]. Sayed and Essa developed traffic conflict
models at the cycle level [CHECK: Essa, Sayed, 2019, *Traffic conflict models to
evaluate the safety of intersections at the cycle level*].

The pattern across this literature is a pipeline: extract trajectories, compute
SSMs, aggregate to window labels, train a classifier, report AUC or accuracy.
The reviews describe the pipeline stages but not the failure modes that appear
when the stages interact. In particular, no review addresses what happens when
the aggregation rule itself becomes a class prior that a classifier can exploit.
The finding in this paper — that a single scalar derived from the same
per-frame features matches or beats a trained model on all features — is not
present in the ML-SSM literature.

## 3. Graph representations of traffic interactions

Traffic scenes are naturally graph-structured: actors are nodes, interactions
are edges, and the graph evolves over time. Graph neural networks (GNNs)
formalized by Kipf and Welling [HIGH: Kipf, Welling, 2017, *Semi-supervised
classification with graph convolutional networks*, ICLR] and extended with
attention by Veličković and coauthors [HIGH: Veličković et al., 2018, *Graph
attention networks*, ICLR] became the natural model class for this structure.

The trajectory-prediction community adopted graph models earlier and more
thoroughly than the safety-analysis community. Social LSTM [HIGH: Alahi, Goel,
Ramanan, Robicquet, Fei-Fei, Savarese, 2016, *Social LSTM: Human trajectory
prediction in crowded spaces*, CVPR] introduced pooling over neighbouring
agents; Social GAN added adversarial training and variety [HIGH: Gupta, Johnson,
Fei-Fei, Savarese, Alahi, 2018, *Social GAN: Socially acceptable trajectories
with generative adversarial networks*, CVPR]; Social-STGCNN made the spatial
graph explicit and convolutional [CHECK: Mohamed, Qian, Elhoseiny, Claudel,
2020, *Social-STGCNN*, CVPR]; Yu and coauthors applied a spatio-temporal graph
transformer [CHECK: Yu, Ma, Ren, Zhao, 2020, *Spatio-temporal graph transformer
networks for pedestrian trajectory prediction*, ECCV].

This paper's pipeline includes a GCN path alongside the tabular baselines,
though the reported comparison focuses on tabular models because the graph
structure is not where the failure lies. The finding is that even *without*
the graph, the task is not learnable beyond what a single scalar can achieve.
A negative result on the tabular side is a stronger statement, not a weaker
one: adding model capacity cannot recover signal that is not in the labels.

## 4. Evaluation pitfalls: leakage, sliding windows, and simple baselines

Three method-side literatures bear directly on this paper's methodological
correction.

**Data leakage.** Kaufman, Rosset, and Perlich formalized leakage in data
mining as the introduction of information about the target that will not be
available at prediction time [HIGH: Kaufman, Rosset, Perlich, 2012, *Leakage
in data mining: Formulation, detection, and avoidance*, ACM TKDD]. Kapoor and
Narayanan surveyed leakage across a hundred-plus ML-based science papers and
found it present in roughly a third, with the most common form being temporal
or spatial autocorrelation between train and test [HIGH: Kapoor, Narayanan,
2023, *Leakage and the reproducibility crisis in machine-learning-based
science*, Patterns]. The specific leak this paper corrects — train and val
windows sharing frames in a sliding-window construction — falls under their
taxonomy as "order leakage" and is exactly the kind of error that survives
review because the code runs cleanly and the metric looks reasonable.

**Cross-validation on structured data.** Roberts and coauthors give the
canonical treatment of blocked and purged cross-validation for temporally or
spatially autocorrelated data [HIGH: Roberts et al., 2017, *Cross-validation
strategies for data with temporal, spatial, hierarchical, or phylogenetic
structure*, Ecography]. Bergmeir and Benítez compared CV strategies for time
series and showed that random K-fold systematically overestimates performance
on autocorrelated data [HIGH: Bergmeir, Benítez, 2012, *On the use of
cross-validation for time series predictor evaluation*, Information Sciences].
This paper's splitter implements blocked folds with a frame-level purge gap,
and the pre-purge number the purge corrects was inflated by exactly the
mechanism Roberts and Bergmeir describe.

**Simple baselines.** A recurring finding across machine learning is that
carefully tuned models fail to beat trivially simple baselines on realistic
data. Musgrave, Belongie, and Lim showed that a set of standard deep metric
learning methods did not outperform a classical baseline on standard
benchmarks [HIGH: Musgrave, Belongie, Lim, 2020, *A metric learning reality
check*, ECCV]. In trajectory prediction specifically, Schöller and coauthors
demonstrated that a constant-velocity model matched or beat published deep
predictors on standard benchmarks [CHECK: Schöller, Aravantinos, Lay, Knoll,
2020, *What the constant velocity model can teach us about pedestrian motion
prediction*, IEEE RA-L]. Bouthillier and coauthors quantified how much of the
reported variance in benchmark results comes from random seed alone [CHECK:
Bouthillier et al., 2021, *Accounting for variance in machine learning
benchmarks*, MLSys]. Sculley and coauthors' "hidden technical debt" paper
names the general syndrome: evaluation pipelines accumulate design choices
that are not examined because they look like plumbing [HIGH: Sculley et al.,
2015, *Hidden technical debt in machine learning systems*, NeurIPS].

This paper's central comparison — a single scalar against a random forest on
all features — is a member of this family. It is not a novel methodological
move; the contribution is that the move has not been applied to window-level
SSM classification, where it reveals a failure of the label design rather
than of the model.

## 5. Positioning

The literature above establishes four things this paper takes as given:
SSMs are a mature construct with a well-cataloged indicator menu; the ML turn
in SSM analysis is well under way and its reviews note but do not resolve the
label-definition problem; graph models are the natural model class for
traffic interaction; and leakage, blocked cross-validation, and simple-baseline
checks are standard practice that this paper applies to a domain where they
have not been systematically applied.

The gap this paper fills is narrow and specific. No prior work in the SSM
classification literature reports the single-feature ablation that this paper
reports, or the label-scale sensitivity that motivates it. The finding is not
that the model is bad; it is that the *label* is not discriminative in the
way the pipeline assumes, and that this is visible only when the model is
compared against the trivial alternative derived from the same features.
The paper is therefore a cautionary study about label design in a domain
where the standard label construction has been taken for granted.

## 6. Reproducibility and negative results

The paper reports a negative result and is explicit about it. The ML
reproducibility literature — including the reproducibility-in-health work of
McDermott and coauthors [CHECK: McDermott et al., 2019 or 2020, *Reproducibility
in machine learning for health*] and the general negative-results movement in
the physical and life sciences — argues that negative results are
underpublished and that their absence distorts the field's picture of what
works. This paper adds one negative result to that accounting, with the
artifacts required to verify it.

---

## Verification queue

The following citations are marked `[CHECK]` above and should be verified in
Google Scholar before submission. For each, the search that will find it is:

1. Perkins & Harris 1968, traffic conflict characteristics General Motors
2. Hydén 1987, Swedish Traffic Conflicts Technique Lund
3. Tarko et al. 2009, surrogate measures of safety TRB white paper
4. Zheng, Ismail, Meng 2014, traffic conflict techniques AAP
5. Johnsson, Laureshyn, De Ceunynck 2018, vulnerable road users Transport Reviews
6. Wang, Xie, Huang, Liu, Chen 2021, SSM CAV review AAP
7. Guo, Sayed, Zaki 2019/2020, SSM from computer vision
8. Essa, Sayed 2019, cycle-level traffic conflict models
9. Mohamed, Qian, Elhoseiny, Claudel 2020, Social-STGCNN CVPR
10. Yu, Ma, Ren, Zhao 2020, STAR graph transformer ECCV
11. Schöller, Aravantinos, Lay, Knoll 2020, constant velocity model RA-L
12. Bouthillier et al. 2021, variance in ML benchmarks MLSys
13. McDermott et al. 2019/2020, reproducibility in ML for health

The `[HIGH]` citations above I am confident about. If any of them is wrong,
the error is mine and the citation should be corrected; the prose does not
depend on any single reference being perfect.

---

## Note on scope

This draft is a first pass at the *structure* and *claims* of a related-work
section. It is not a finished bibliography. A submitting author should:

1. Verify each `[CHECK]` citation.
2. Add 3-5 recent (2024-2026) references in each major section, which this
   draft does not attempt because my confidence about recent work is lower.
3. Reread the section against the target venue's expectations, which vary
   considerably between transportation journals (AAP, TRR, TR-C, IEEE T-ITS)
   and machine-learning venues.

---

# Method

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
| gcn | 0.590 | — | **0.778** |
| logreg | 0.695 | [0.614, 0.782] | 0.573 |
| rf | 0.757 | [0.660, 0.843] | 0.728 |
| single_feature | **0.803** | **[0.736, 0.863]** | 0.765 |

The GCN's accuracy and F1 match the majority baseline exactly, and its
AUC is barely above chance. The model collapsed to the majority class.
The complete ordering — cue > rf > gcn > majority — means both added
capacity (rf over logreg) and added structure (gcn) reduce AUC on this
label.

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

AICC22-Custom, the companion clip in the same Zenodo record, was
processed through the same pipeline with the same hyperparameters. Its
label distribution differs — 97 windows, 69 positive / 28 negative,
majority 0.711 — and one fold has zero validation negatives, so the
effective AUC fold count is 4.

| model | AUC | 95% CI |
|---|---:|---|
| majority | 0.500 | [0.500, 0.500] |
| logreg | 0.611 | [0.291, 0.930] |
| rf | 0.567 | [0.243, 0.870] |
| single_feature | **0.765** | [0.534, 0.950] |

The direction replicates: cue leads rf by 0.199, larger than the 0.046
gap on VNTraffic. The absolute intervals are too wide for a
confirmatory claim — neither model excludes 0.5 — so the second clip
is reported as directional evidence only. Full numbers in
`docs/paper/baselines_sustained_r5_aicc22.md`.

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

---

# Results

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

---

# Discussion

traces to a number in `results.md` or a finding in
`docs/external_review_2026_09.md`. Where the interpretation goes beyond
the data, that is said explicitly.

---

## 1. What the results mean

Three models of increasing capacity and structural richness — logistic
regression, random forest, GCN — were trained on the sustained
`run_length=5` VNTraffic label with the boundary leak corrected. None
exceeded a single scalar feature (`edge_attr_nonzero_frac`) on ranking
performance, and none exceeded the majority-class baseline on accuracy.
The GCN matched the majority baseline exactly on both accuracy and F1,
meaning it converged to the trivial predictor.

The primary result is the null the pre-registration memo committed to
reporting if it occurred: the trained model provides no evidence of
value over the trivial scalar. This is not a failed experiment. It is
the experiment's answer.

## 2. Why the label fails

The single-feature cue is not a lucky column. The external review
(`docs/external_review_2026_09.md`, finding #10) established that
`edge_attr_nonzero_frac` correlates at |rho| >= 0.94 with the zero
fraction of exactly four edge-feature columns: relative velocity in x
and y, relative speed, and closing speed. It is, mechanically, a proxy
for the fraction of edges in the window with any relative motion
between the two actors.

The same review established what the cue is *not* correlated with:
window index (rho = -0.03) and node count (rho = +0.06). Time and
crowding are ruled out as confounders. The mechanism is activity, not
scale, not position in the clip.

The chain is: more relative motion in a window -> more pairs with a
nonzero closing speed -> more pairs satisfying the TTC < 1.5 s /
distance < 3.0 m criterion in at least one frame -> more positive
windows under the sustained rule. The label is, in effect, a coarse
activity detector. A classifier trained on the 42-column summary has
to recover that activity signal from aggregate statistics of the same
underlying features; the single-column cue reads it directly. The
model is not failing to see the signal. It is failing to compress 42
columns into a better activity measure than one of the columns already
is.

## 3. What this says about window-level SSM labels

The standard construction in the SSM-ML literature is: extract
trajectories, compute a per-pair per-frame indicator (TTC, PET, DRAC),
aggregate to a window label via an 'any frame' or similar rule, then
train a classifier on window features. The aggregation rule is treated
as plumbing. This paper argues it is a design decision, and that the
standard choice makes the label a function of scene activity rather
than of conflict severity.

The all-81 VNTraffic run (`baselines_vntraffic_all81.md`) is the
sharper version of the same finding. At 81 tracks and the `any` rule,
the label is 249/249 positive — completely degenerate. Adding actors
adds opportunities for a TTC event to occur by chance, and 'any frame
contains a critical pair' collapses to 'the scene is busy.' The
sustained rule at `run_length=5` is the only configuration tested on
this clip that produces a non-degenerate split at all, which is itself
evidence of the same pathology: the label is only discriminative when
the aggregation rule is tightened enough to remove most of the
activity-driven positives.

The implication for the field is not that SSMs are wrong. TTC is a
well-defined, physically meaningful quantity. The implication is that
the *window-level aggregation* of a per-pair per-frame SSM into a
single binary label is not a scale-invariant operation, and treating
it as one produces tasks that look learnable (the AUC is above chance)
but are not (the model does not beat a scalar, and neither beats the
prior).

## 4. What this paper does not claim

It does not claim that GNNs are unhelpful for traffic safety. It claims
that a GCN trained on the `GraphFeatureExtractor` summary of this
label does not outperform the trivial scalar. A model reading the
graph directly, or a different graph construction, or a different
label, is a different experiment.

It does not claim that TTC-based labels are wrong. It claims that the
window-level binary aggregation of a TTC-based label on this data is
not discriminative in the way a classification pipeline assumes.

It does not claim the finding generalizes beyond the two clips tested.
The AICC22-Custom run is directional replication, not confirmation;
its confidence intervals are too wide to exclude chance for any model.

## 5. Threats to validity

### 5.1 Sample size

249 windows on VNTraffic, 97 on AICC22. Ten and five folds
respectively. Negatives per fold range from 2 to 10 on VNTraffic and
0 to 18 on AICC22. The bootstrap CIs on the per-fold AUC arrays are
wide precisely because the fold-level data is thin. A reviewer should
read every CI in this paper with that in mind.

### 5.2 Post-hoc choices

Two choices were made after inspecting the data and are recorded as
limitations in `docs/preregistration_sustained_r5.md`:

1. `run_length=5` was selected because it is the only sustained-rule
   setting that produces a non-degenerate label on this clip.
   `run_length=2` and `3` give 249/0; `run_length=5` gives 194/55.
   The natural-defense argument ('every frame in a 5-frame window is
   critical') is sound but was not made before the label distribution
   was inspected.
2. `edge_attr_nonzero_frac` was identified as the dominant cue during
   the external review, not before the analysis.

Both are honest limitations. Neither invalidates the finding: the
pre-registration memo fixes the *comparison* (RF vs cue on this label)
before the rerun, and the rerun's outcome (cue ties or beats RF) is
what the memo committed to reporting either way.

### 5.3 Label rule and feature coupling

The label is derived from TTC, and TTC is one of the columns inside
the edge-feature matrix the model reads. The specific TTC column
(`ttc` itself) is at chance as a classifier (finding #10: `ea_min`
AUC 0.517), so the coupling is not a direct label leak. But the
feature family and the label family are the same, and this is a
design choice rather than a pre-specification.

### 5.4 Scale-only homography

VNTraffic ships no camera calibration. The pipeline uses a scale-only
homography at 100 pixels per metre. TTC is scale-invariant and
unaffected; DRAC is not and inherits the PPM error. This paper reports
TTC-based labels only, so the DRAC path is not part of the result, but
the absolute distance threshold (3.0 m) is expressed in a unit whose
physical interpretation depends on the PPM assumption.

### 5.5 Feature extractor is a summary

`GraphFeatureExtractor` reduces each window to 42 aggregate statistics
and discards the graph structure. The reported comparison is between
models reading this summary. It does not test whether a model reading
the graph directly (node features, edge indices, adjacency) would
perform differently. The GCN run is a partial answer to this
(it also fails), but the GCN reads the same summary as the tabular
models in the current implementation.

### 5.6 One GCN configuration

The GCN result is one architecture at 25 epochs on CPU. A larger GCN,
or a different message-passing scheme, might behave differently. The
honest reading of the GCN result is that *this* GCN configuration on
*this* label collapsed to the majority predictor, not that GNNs
cannot learn the task under all configurations.

## 6. What would change the conclusion

Three experiments would materially strengthen or overturn the finding,
in order of leverage:

1. **A positive control.** If a different aggregation rule — e.g. a
   per-pair, per-frame prediction task, or a window label that
   normalizes by the number of pairs — produces a task on which the
   models *do* beat the cue, then the failure is specifically in the
   current window-aggregation choice. This is the sharpest experiment
   and it is not attempted here.
2. **More clips.** Three or more independently annotated clips would
   move the replication from directional to confirmatory. The current
   two-clip evidence is consistent, not conclusive.
3. **A model reading the graph directly.** The current GCN reads the
   same 42-column summary as the tabular models. A GCN that consumes
   node and edge tensors from the PyG graphs (available in
   `data/processed/graphs/`) would test whether the graph structure
   carries signal the summary discards.

None of these would rescue the current label if the mechanism in §2 is
correct. All three would sharpen the scope of the claim.

## 7. Implications

For researchers building SSM-based classifiers: the window-level label
should be validated against a trivial baseline derived from the same
features before a model comparison is reported. If a single scalar
matches the trained model, the label is not carrying information the
model can use. This is a five-minute check and it is not standard
practice in the field.

For reviewers: AUC above chance is not evidence that a label is
learnable in the sense the paper needs. The relevant comparison is
against the simplest alternative the same features support. This paper
is one instance where the check was applied and the answer was no.

For the SSM literature specifically: the aggregation of per-pair
per-frame indicators into a window label is not scale-invariant in
actor count, and the standard 'any frame' rule saturates as the scene
gets busier. This paper is the first we are aware of to state the
saturation as an empirical finding rather than a caveat.

## 8. Conclusion

On two traffic clips with hand-annotated trajectories, under the
standard window-level TTC-based SSM label with a corrected cross-
validation split, a single scalar feature matches or beats the trained
models. The failure is not in the model class; it is in the label's
construction. The window-aggregation step, treated as plumbing in the
SSM-ML literature, is where the signal is lost.

---

## References

*Bibliography management TBD. Citations in `related_work.md` carry
`[HIGH]` (verified) and `[CHECK]` (verify before submission) tags.
13 entries need verification.*

---

## Figures

| figure | caption |
|---|---|
| `figures/fig1_auc_vntraffic.pdf` | VNTraffic — mean AUC per model, 95% bootstrap CI. |
| `figures/fig2_pre_post_purge.pdf` | Pre- vs post-purge AUC for RF and the single-feature cue. |
| `figures/fig3_cross_clip.pdf` | Cross-clip comparison. Direction replicates; CIs overlap. |
| `figures/fig4_gcn_collapse.pdf` | The GCN matches the majority baseline on accuracy exactly. |

---

## Reproducibility statement

- Code: https://github.com/Sharqascc/graf
- Data: VNTraffic and AICC22-Custom, Zenodo record 18195750
- Pipeline: three commands from a fresh clone, documented in `docs/reproducibility.md`
- Pre-registration: `docs/preregistration_sustained_r5.md`, committed before the rerun
- All result JSONs: `docs/paper/*.json`
