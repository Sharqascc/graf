# Research direction: window -> pair-level surrogate safety

**Working reference. Anything not on this page is distraction.
Locked 2026-11-09, supersedes 2026-11-07 version.**

## Pivot (refined)

**Old question:** is the window-level TTC label broken?

**New question:** do window-level TTC labels measure **safety risk** or
**traffic exposure**?

The reframe is a claim about the *aggregation rule*, not about TTC
values being wrong. Under `any` aggregation, per-pair event probability
p and eligible-pair count N produce:

    P(window labelled dangerous) = 1 - (1 - p)^N

Same per-pair risk, more eligible pairs, higher label rate. That is the
mechanism we are testing. TTC itself can be perfectly correct; the bias
lives in the `at least one event` aggregation.

## Pre-registered interpretations (locked before SCI run completes)

Run on SCI (2700 frames, heterogeneous traffic, ~18 actors per frame).

| SCI result | interpretation |
|---|---|
| All labels 1.000, no threshold fixes it | saturation confirmed on this scene; case study finding |
| Some threshold yields 0.2 <= rate <= 0.8 | use that variant for regression; test below |
| exposure-only AUC > behavior-only AUC | label is exposure counter; thesis supported |
| behavior-only AUC > exposure-only AUC | label captures behavior; thesis falsified here |
| permutation-null rate approx. equals real rate | label is opportunity-count, not interaction-detector |
| permutation-null rate < real rate | real interactions matter even at saturation |

**This block is the commitment. No re-interpretation after seeing
numbers.** If the answer is falsification, it is reported as
falsification.

## RQs (final)

- **RQ1 saturation** — how do TTC threshold, run-length requirement, and
  window length affect positive rate? (target: find settings where rate
  is not 0 or 1)
- **RQ2 exposure dependence** — does per-pair opportunity count
  (eligible interacting pairs, pair-seconds) explain window labels
  better than per-pair behavior does?
- **RQ3 safety validity** — deferred. Needs independent labels. Out of
  scope for this paper.

## The three tests

**T1 eligible-pair count.** For each window compute:
- actor count (naive)
- eligible interacting pairs: within X m, closing rate above threshold,
  plausible intersection of projected paths
- pair-seconds (eligible pairs x frames within distance)

Regress `label ~ T1 features` alone, then `label ~ T1 + behavior`. Compare
coefficients. Exposure dominates -> supports mechanism.

**T2 behavior-only baseline.** Same regression but behavior features
only (min TTC, mean TTC, class fractions). Compare to T1. If T2 beats
T1 -> falsification.

**T3 permutation null.** Time-shift each track independently by a random
offset (uniform over the clip). Density preserved; real spatial
interactions destroyed. Recompute label. Compare rate to real.

If null_rate approx. real_rate -> label measures opportunity, not
interactions. If null_rate much lower -> real interactions matter.

## Scope (honest)

- **One site.** SCI is a case study, not a confirmation of the field.
- **No independent ground truth.** Everything is TTC-derived. The
  permutation test is the closest thing to independent evidence we can
  do without human labels.
- **Claim ceiling.** "On this heterogeneous site, window labels under
  `any` aggregation measure exposure as much as per-pair risk." Not
  "the field is broken."

## Success criterion

**Paper (methods + case study):** "Empirical evidence that window-level
`any` aggregation of TTC events conflates interaction risk with
opportunity count, on heterogeneous traffic video."

Target venues: SIGSPATIAL workshop / TRB / AAA short paper. Not a full
conference paper without a second site.

## Execution order

1. Finish SCI artifact regeneration (GitHub run #42638434164)
2. Extend `scripts/sci_density_analysis.py`: add T1 features
   (eligible-pair count, pair-seconds) and threshold x window sweep
3. Run on SCI. Report all 8 label variants + sweep grid
4. Run T1 regression on whichever variant is non-degenerate
5. Run T3 permutation null on same variant
6. Pre-register T2 result interpretation
7. Write up: results table + method + honest scope

## Data status (2026-11-09)

- VNTraffic: not available (removed from scope — homogeneous)
- AICC22: not available
- SCI: GitHub artifact re-run in progress (90 min, id 42638434164)
- Own footage: unknown
- Human labels: none

## Not doing

- GETAD architecture (wrong shape for the task)
- Standalone PET paper (one option, not the fix)
- Full-model ML training until the label question is settled
- Claiming field-wide validation from one site
- Post-hoc story fitting after seeing SCI numbers
- Architecture changes to `sci_density_analysis.py` beyond T1 + sweep

## Files

- `docs/research/DIRECTION.md` - this file
- `scripts/density_diagnostic.py` - VNTraffic variant (retained, CI-gated)
- `scripts/sci_density_analysis.py` - SCI variant (CI-gated)
- `scripts/rerun_paper_results.py` - B.1 re-run helper (CI-gated)
- `docs/research/sci_density/` - SCI output (to be created)
