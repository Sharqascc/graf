# Baseline comparison — VNTraffic

**Status:** Declared-setup run. Not tuned, not cherry-picked.

## Setup

- **Tracks:** top-10 longest of 85 (VNTraffic clip)
- **Label source:** TTC (window positive if any frame has 0 < TTC ≤ 1.5 s within 3.0 m)
- **CV:** 5-fold, blocked (contiguous frame-ordered folds)
- **Windows:** window_size=5, stride=2 → 249 windows
- **Class balance:** 220 positive / 29 negative (majority baseline 0.884)
- **Epochs:** 25, batch_size 16, lr 0.001, pos_weight 2.0

## Results

| model | mean acc ± std | mean F1 | mean AUC ± std | neg/fold | pooled acc | runtime (s) |
|---|---|---|---|---|---|---|
| majority | 0.882 ± 0.068 | 0.936 | 0.500 ± 0.000 | 11/3/8/5/2 | 0.884 | 2.14 |
| logreg | 0.603 ± 0.270 | 0.673 | 0.608 ± 0.226 | 11/3/8/5/2 | 0.594 | 2.85 |
| rf | 0.875 ± 0.061 | 0.932 | 0.481 ± 0.216 | 11/3/8/5/2 | 0.876 | 5.65 |
| mlp | 0.882 ± 0.068 | 0.936 | 0.494 ± 0.292 | 11/3/8/5/2 | 0.884 | 2.39 |
| gcn | 0.882 ± 0.068 | 0.936 | 0.664 ± 0.213 | 11/3/8/5/2 | 0.884 | 279.12 |

## Reading

**The label distribution is the dominant fact.** 29 negatives across 249 windows means each blocked fold sees only a handful of negatives ([11, 3, 8, 5, 2] per fold in this run). A single misclassified negative shifts per-fold AUC by 0.1–0.4, which is why every model's mean AUC has a std of 0.2–0.3.

**No model beats the majority baseline on accuracy.** Every model either matches the majority rate (0.882) or falls below it (logreg at 0.603). On AUC, no model's mean is statistically distinguishable from chance at n=5 folds (minimum attainable two-sided Wilcoxon p ≈ 0.0625).

**Pooled AUC is not reported.** Training a separate model per fold means fold models produce scores on different calibration scales; ranking one fold's positives against another fold's negatives is not a valid AUC computation. The `pooled_auc_DEBUG_DO_NOT_REPORT` field is retained in the JSON only for debugging.

## What this does and does not establish

**Establishes:** With this label definition and this clip, the classification task is nearly degenerate (88% positives). No model — classical or graph — separates the positive and negative windows at a rate that a small-sample statistical test can confirm.

**Does not establish:** That the GNN architecture is unhelpful, or that the pipeline is wrong. Both conclusions would require a task with balanced classes and enough negatives to estimate AUC with reasonable variance. See the label-sensitivity analysis (forthcoming) for whether any parameterization of the TTC-based label produces such a task.
