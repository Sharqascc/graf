# Abstract and Introduction

**Status:** first draft, written last so it points at finished findings.
Every number cited here is sourced from `results.md`; every mechanism
from `discussion.md`. No new claims.

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
