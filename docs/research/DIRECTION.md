# Research direction: window -> pair-level surrogate safety

**Working reference. Anything not on this page is distraction.
Locked 2026-11-07.**

## Pivot
Current question: is this 5-second window dangerous?
New question: will this specific pair of road users develop a critical
conflict?
Unit of analysis: window -> dyad. Only change all four advisor reviews
agree on.

## Plan

**P1 (this week) - diagnostic.** Does the random forest still beat the
majority baseline inside each density tercile? Run
`scripts/density_diagnostic.py` on VNTraffic and AICC22.
- AUC <= 0.55 in every bin -> diagnosis confirmed -> P2a.
- AUC >= 0.65 in any bin -> signal survives density -> P2b.
- else -> inspect per-bin before choosing.

**P2a (if confirmed).** Pair-level relabel: density-adjusted excess
risk OR pair-level binary TTC. Re-run all baselines. Models should now
beat the density baseline on the new label.

**P2b (if signal survives).** Soften thesis to "window labels are
noisy, not useless." Different paper, different pitch.

**P3 (if P2a lands AND reviewers push back).** Human near-miss labels
on 200-300 windows. Requires footage check first.

## Data status (2026-11-07)
- VNTraffic, AICC22: available locally, gitignored, not redistributable.
- Own footage: **unknown** - P3 blocked until confirmed.
- Human labels: **none yet**.

## Not doing
- GETAD architecture (GAT + Transformer + GPE + CW-NLL) - wrong shape
  for pair-level classification; no advisor recommends it.
- PET as standalone paper - one pair-level option, not the fix itself.
- Positive-result-only reframe - the diagnosis is the foundation.
- Abandoning GRAF, tracking, or graph pipeline - all reused in P2a.

## Success criterion
Paper 1: "the window-level TTC label is a density proxy" (P1 alone).
Paper 2 (optional): "pair-level fix that works" (P1 + P2a).

## Files
- `docs/research/DIRECTION.md` - this file.
- `scripts/density_diagnostic.py` - P1 tool (next commit).
- `docs/research/density_results.md` - P1 output (after run).
