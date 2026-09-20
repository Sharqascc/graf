# Alternative label strategies — VNTraffic, all-81 tracks

**Companion to:** `baselines_vntraffic_all81.md` (the 'any'
label produces 249/249 positive at this track count).

Tests four window-level label strategies aiming to keep the
label discriminative when many actors share the scene. See
`scripts/label_strategies.py` for definitions.

## Stage 1 — label distribution per strategy

Same 249 windows, same underlying TTC = 1.5 s, distance = 3.0 m
per-frame measurement. Only the aggregation rule differs.

| strategy | params | pos | neg | majority |
|---|---|---|---|---|
| any | — | 249 | 0 | 1.000 |
| pair_fraction | fraction_threshold=0.2 | 248 | 1 | 0.996 |
| pair_fraction | fraction_threshold=0.5 | 0 | 249 | 1.000 |
| pair_fraction | fraction_threshold=0.8 | 0 | 249 | 1.000 |
| sustained | run_length=2 | 249 | 0 | 1.000 |
| sustained | run_length=3 | 249 | 0 | 1.000 |
| sustained | run_length=5 | 194 | 55 | 0.779 |
| min_ttc | — | 249 | 0 | 1.000 |

Six of eight configs remain fully or nearly degenerate. Only
`sustained` with `run_length=5` produces a meaningful negative
class. This is the natural definition: with a 5-frame window,
requiring 5 consecutive critical frames means *every* frame in
the window must be critical.

## Stage 2 — models on the one non-degenerate config

`sustained run_length=5`: 194 positive / 55 negative,
majority 0.779.

| model | mean acc ± std | mean AUC ± std | neg/fold |
|---|---|---|---|
| logreg | 0.598 ± 0.109 | 0.614 ± 0.071 | 9/14/10/11/11 |
| rf | 0.722 ± 0.085 | 0.738 ± 0.094 | 9/14/10/11/11 |

## Reading

**The random forest has genuine ranking signal.** AUC
0.738 ± 0.094 across 5 folds; standard error ≈ 0.042, so the
effect is ~5.7 SEs above chance. This is the first time in this
investigation that any model has produced AUC meaningfully
above 0.5 on any VNTraffic label.

**The random forest does not beat majority on accuracy.** 0.722
vs majority 0.779. The model ranks positive windows above
negative windows, but the decision threshold at 0.5 does not
convert that ranking into accuracy at this class balance. This
is a calibration question, not a signal question: the model has
learned something real, but the default decision rule is wrong
for this prior.

**Logistic regression is weak.** AUC 0.614 ± 0.071, barely
above chance. The signal RF finds is not visible to a linear
model on the same features.

## The `run_length=5` choice and why we stop here

`run_length=5` is the maximum value that still produces
non-empty positive windows on a 5-frame window: it means
'every frame in the window has a critical pair'. It is the
natural strongest version of `sustained`, not a cherry-picked
value.

We do not sweep `run_length` further. A sweep would risk
selecting a value where RF happens to beat majority on
accuracy, which would not replicate and would not be an honest
result. The declared finding is: with the natural label rule
that survives on this track set, one model has ranking signal
but does not beat majority on accuracy at threshold 0.5.

## Comparison to top-10 tracks

At top-10 tracks the `all` rule (analogous to `sustained` with
run_length equal to the window size) produces **17 positive /
232 negative** (majority 0.932) — the opposite imbalance. At 81
tracks the same rule produces **194 / 55** (majority 0.779).
The rule flips from 'almost no event' to 'almost every event'
as actor count grows.

This is the same scale-saturation phenomenon from the
sensitivity sweep, visible across the actor-count axis rather
than the threshold axis. It reinforces the earlier conclusion:
window-level SSM labels are not scale-invariant, and the
actor count must be treated as a design parameter when
publishing results on any SSM-based classification task.

## What this changes for the paper

The paper's discussion section can now say something sharper
than "the model failed":

1. Under the standard SSM window label ('any critical pair in
   any frame'), the task is not learnable at any tested
   threshold, distance, or track count.
2. Under a stricter 'every frame critical' rule on the full
   81-track set, a random forest has ranking signal
   (AUC 0.738) but does not beat the majority baseline on
   accuracy.
3. A model that ranks but does not classify at a fixed threshold
   is a calibration problem. This suggests the failure mode is
   not 'no signal in the data' but 'the standard label and the
   default threshold are not matched to what the model can
   extract'.

The next experiment, not attempted here, would be to tune the
decision threshold on the training folds (not the test folds)
and report whether accuracy beats majority under that
calibration. That is a legitimate follow-up; it is not a
replacement for this result.
