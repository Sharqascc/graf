# Superseded: pre-B.1 results

Every label-bearing artifact here was generated under the TCPA-only
definition (pre-B.1) in docs/audit/ttc_formula_decision.md. That
definition fires critical on pairs that approach within
distance_threshold but never get closer than dcpa_threshold.

On 2026-11-07 the project adopted B.1 (DCPA-gated TCPA). Every
label-bearing artifact here is stale until regenerated.

Affected: any file here reporting a label count, positive rate, model
accuracy, F1, AUC, confusion matrix, or baseline comparison.

Regenerate:
  python scripts/compare_baselines.py --label-strategy any \\
      --ttc-dcpa-threshold 1.5 --ttc-threshold-seconds 1.5 \\
      --ttc-distance-threshold 3.0

Until a commit with message 'rerun labels under B.1' touches every
*.json here on main, treat every number here as pre-B.1.
