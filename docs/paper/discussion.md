# Discussion

**Status:** first draft. Interpretive section; every factual claim
traces to a number in `results.md` or a finding in
`docs/external_review_2026_09.md`. Where the interpretation goes beyond
the data, that is said explicitly.

---

## 1. What the results mean

Three models of increasing capacity and structural richness — logistic
regression, random forest, GCN — were trained on the sustained
`run_length=5` VNTraffic label with the boundary leak corrected. No
trained model was detected to outperform a single scalar feature
(`edge_attr_nonzero_frac`) on ranking performance, and none exceeded
the majority-class baseline on accuracy. The GCN's accuracy matched
the majority baseline exactly; its AUC did not establish above-chance
ranking at this fold count.

The primary result is the null the pre-registration memo committed to
reporting if it occurred: no detectable added value over the
single-feature cue (paired $\Delta$AUC = −0.046, corrected 95% CI
[−0.138, +0.046], $p=0.29$). This is not a failed experiment. It is
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
single binary label is not a scale-invariant operation. On this data
it produces a task with real ranking signal — the cue and the random
forest both beat chance at corrected $p<0.01$ — but where a 42-feature
model provides no detectable added value over one of its own columns,
and where ranking does not convert to classification above the class
prior at threshold 0.5.

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
validation split, no trained model was detected to outperform a
single scalar feature. The cue and the random forest both carry
statistically significant ranking signal, but a 42-feature model adds
nothing measurable over one of its columns, and neither beats the
majority-class accuracy at threshold 0.5. The window-aggregation step,
treated as plumbing in the SSM-ML literature, is where the added value
is lost.
