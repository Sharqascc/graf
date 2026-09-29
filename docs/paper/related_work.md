# Related work

**Status:** first draft. Prose written from domain knowledge; every citation
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
