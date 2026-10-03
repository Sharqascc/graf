# Statistical analysis

**Status:** methodology companion to `results.md`. Every number here is
computed from the per-fold AUC arrays in `docs/paper/baselines_*.json`.
Corrects the independent-CI-overlap comparison used in the first draft
of `results.md` with a paired-difference analysis.

---

## 1. Why the first-draft comparison was wrong

The first draft of `results.md` compared the random forest and the
single-feature cue by asking whether their independent 95% confidence
intervals overlapped. Overlap of independent CIs is not a test of
difference: two intervals can overlap while the underlying quantities
differ significantly, and two intervals can fail to overlap while the
difference is not distinguishable from zero. The correct comparison
for two models evaluated on identical folds is a **paired** analysis
on the per-fold difference:

$$\Delta_i = \mathrm{AUC}_{A,i} - \mathrm{AUC}_{B,i}$$

which eliminates the cross-fold covariance that the independent-CI
comparison ignores. This section reports the paired analysis and
replaces the overlapping-CI claim in `results.md` §2 and §6.

## 2. Method

**Corrected resampled t-test (Nadeau & Bengio, 2003).** Cross-validation
folds share training data, so fold-level metrics are not independent.
The correction inflates the standard error of the mean difference by
a factor of $\sqrt{1/n + 1/(k-1)}$ where $k$ is the fold count. At
$k=10$ this is an inflation of $\sqrt{1 + n/(k-1)} \approx 1.45$
relative to the naive paired SE. Without the correction the paired
p-values are anti-conservative.

**Exact sign-flip permutation test.** For each paired comparison the
mean of $\pm d_i$ across all $2^n$ sign assignments gives an exact
null distribution of the mean under symmetry of the differences. This
is a nonparametric cross-check on the corrected t-test; it does not
itself correct for fold dependence and is reported as a sanity check
only.

**Effect size.** Cohen's $d_z = \bar\Delta / s_\Delta$ is reported for
each comparison. At $n=10$ or fewer folds it is unstable and should be
read as a magnitude indicator rather than a precise estimate.

**Wilcoxon signed-rank.** Not used as a primary test here. For a
two-sided Wilcoxon on $k$ nonzero pairs the minimum attainable $p$ is
$2^{1-k}$: $0.0625$ at $k=5$ and $0.125$ at $k=4$. Wilcoxon is
therefore structurally unable to reach $p<0.05$ on the AICC22 dataset
and cannot be the basis of any claim at that fold count.

**Multiplicity.** Only RF vs cue on VNTraffic was pre-specified in
`docs/preregistration_sustained_r5.md`. The remaining comparisons are
exploratory and are labeled as such. Where a family of tests is
reported together, a Holm correction is applied to the exploratory
family and both raw and adjusted $p$ values are shown.

## 3. VNTraffic ($k=10$) — absolute performance vs chance ($AUC = 0.5$)

One-sample corrected t on the per-fold AUC minus 0.5.

| model | mean AUC | corrected 95% CI | corrected $p$ vs 0.5 | verdict |
|---|---:|---|---:|---|
| single-feature cue | 0.803 | [0.690, 0.915] | **0.0002** | beats chance |
| random forest | 0.757 | [0.596, 0.918] | **0.0056** | beats chance |
| logistic regression | 0.695 | [0.614, 0.782] | (see raw JSON) | beats chance (raw $p<0.05$) |
| GCN | 0.590 | [0.382, 0.798] | 0.3542 | does not establish above-chance performance |
| majority | 0.500 | [0.500, 0.500] | 1.0000 | by construction |

The GCN's CI includes 0.5 and its corrected $p$ is 0.35. The correct
statement is that the available folds do not establish above-chance
ranking for the GCN, not that the GCN is at chance. Its point estimate
is 0.590 and its CI upper bound reaches the random forest's point
estimate.

## 4. VNTraffic — paired differences

| comparison | $\bar\Delta$ AUC | corrected 95% CI | $p$ (corr. $t$) | $p$ (perm.) | $d_z$ | reading |
|---|---:|---|---:|---:|---:|---|
| RF − cue *(primary)* | −0.046 | [−0.138, +0.046] | 0.2862 | 0.1406 | −0.52 | no detectable difference |
| GCN − cue | −0.213 | [−0.433, +0.006] | 0.0557 | 0.0195 | −1.01 | borderline; not significant at α=0.05 |
| GCN − RF | −0.167 | [−0.446, +0.112] | 0.2083 | 0.0762 | −0.62 | no detectable difference |
| RF − majority | +0.257 | [+0.096, +0.418] | 0.0056 | 0.0039 | +1.66 | RF beats a constant predictor |

The primary comparison (RF − cue) yields a CI that spans zero. The
pre-registered decision rule in `docs/preregistration_sustained_r5.md`
is retained verbatim in `results.md` §6 and fires its outcome 1 on the
pre-purge independent-CI framing. The paired analysis reported here is
a post-hoc supplement that sharpens the same conclusion: no detectable
difference between the two estimators.

The GCN − cue comparison is borderline. Three methods disagree on
whether it crosses α=0.05: corrected $t$ gives $p=0.056$, the exact
permutation gives $p=0.020$, and the naive paired $t$ gives $p=0.011$.
The corrected $t$ is the appropriate primary test given fold dependence;
the honest reading is that the GCN point estimate is lower than the
cue's, at a level that would be significant under the anti-conservative
naive test and is not under the corrected one.

## 5. AICC22-Custom ($k=4$) — descriptive only

One fold has zero validation negatives, so effective $n=4$. The
corrected CI on RF − cue is [−1.618, +1.220] — spanning 2.8 AUC units
on a metric bounded in [0, 1]. The bootstrap over four values has
$\binom{8}{4}=70$ distinct resamples; the interval is essentially
uninformative. **No inferential claim is made on this dataset.**

Reported descriptively:

- Per-fold RF − cue differences: −0.944, −0.429, +0.056, +0.522
- Mean ΔAUC: −0.199
- Cue mean AUC: 0.765, RF mean AUC: 0.567

The direction is consistent with VNTraffic. The magnitude is not
interpretable at this fold count.

## 6. Multiple comparisons

Only one comparison was pre-specified. The remaining pairwise tests
in §4 are exploratory and would survive a Holm correction only for the
RF − majority comparison (raw $p=0.0056$, Holm-adjusted $p=0.0168$
over the family of three non-primary tests). The GCN − cue comparison
would not survive Holm (adjusted $p \approx 0.11$). The correct
framing is exploratory for all of §4 except the primary.

## 7. Structural limits of this analysis

- **Fold count.** $k=10$ on VNTraffic, effective $k=4$ on AICC22. The
  VNTraffic analysis is adequately powered to detect the effect sizes
  reported; the AICC22 analysis is not.
- **Unequal fold weight.** Folds with 2 validation negatives and folds
  with 10 are given equal weight in the mean. Per-fold AUC on 2
  negatives has variance an order of magnitude larger than on 10.
- **Approximate correction.** The Nadeau-Bengio correction assumes a
  random partition. Our folds are blocked and purged; the correction
  is approximate for this design and should be read as the
  conservative end of the plausible range.
- **Permutation test floor.** With $n=10$ the sign-flip permutation
  has $2^{10}=1024$ distinct outcomes, floor $p \approx 0.002$.
  Small $p$-values from this test should not be interpreted below that
  floor.

## 8. What changes in the paper text

The following phrases in the first draft of `results.md`, `discussion.md`,
and `introduction.md` were rewritten to match this analysis:

- "matches or beats" → "no statistically detectable difference"
- Independent 95% CIs on each model → paired 95% CIs on the difference
- "neither model beats chance" (where present) → per-model chance tests;
  cue and RF beat chance, GCN does not establish it
- "the GCN collapses to majority" → "the GCN's accuracy equals the
  majority baseline; its AUC does not establish above-chance ranking"

The pre-registered decision rule in `results.md` §6 is preserved
verbatim; the paired analysis is added as a labeled post-hoc supplement.
