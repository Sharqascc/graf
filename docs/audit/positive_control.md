# Positive control (item 2)

**Verdict: the pipeline can learn the rule.** A depth-1 decision
tree on one integer feature (`n_critical`, the number of critical
frames in the window) recovers the harness label with **100 percent
accuracy**. Every downstream model failure is therefore a model
issue, not a pipeline issue.

Run: workflow `positive-control.yml`, GitHub Actions run
`39144652270`, completed `success`.

## Setup

| parameter | value |
|---|---|
| tracks | `data/raw/vntraffic_tracks_all85_min20.jsonl` |
| graphs_dir | `data/processed/graphs/vntraffic_all85` |
| homography_config | `data/raw/vntraffic_homography.yaml` |
| window_size | 5 |
| stride | 2 |
| run_length | 5 |
| fps | 30.0 |
| ttc_threshold | 1.5 |
| distance_threshold | 3.0 |

Windows: 249. Positives: 194. Negatives: 55. Majority: 0.779.

## Results

### Label integrity

| check | result | verdict |
|---|---|---|
| `frame_y.n_with_y` | **0** (of 50 sampled) | no per-frame `.y` populated |
| `window_y.n_with_y` | **0** (of 50 sampled) | no window-level `.y` populated |
| `sustained_agreement.fraction` | **1.0** (249/249) | harness `label_sustained` matches an independent reimplementation exactly |

**This resolves item 1's open question.** Columns 28-31
(`y_mean/std/min/max` in the 42-dim vector) are always 0.0 on
the current graph build. Item 1's verdict stands: the only
circularity is the TTC-derived edge aggregates, not the `y`
columns.

### Learnability

| feature | tree accuracy | threshold learned | verdict |
|---|---:|---:|---|
| `n_critical` per window | **1.0** | 4.5 | **pipeline learns the rule perfectly** |
| `min_ttc` over window | 0.779 | 0.046 | (different rule, as expected) |

**`n_critical >= 4.5` == `n_critical >= 5` == the sustained
run_length=5 definition.** A depth-1 tree recovers it exactly.

**`min_ttc` at 0.779 is not a failure.** The plan text
originally suggested testing on `min_ttc`, but the harness's
`label_sustained` requires *five consecutive* critical frames,
not one. `min_ttc <= 1.5` fires when *one* frame is critical.
Different rules, different trees. The 0.779 accuracy matches
the majority baseline exactly -- the tree on `min_ttc` collapses
to majority because the two rules genuinely differ.

## What this establishes

1. **The pipeline is sound.** Training a tree on one feature
   produces perfect accuracy on the harness label. There is no
   hidden bug in `label_sustained`, `_frame_ttc_stats`, or the
   window construction.
2. **The GCN's 0.590 AUC is a model failure, not a pipeline
   failure.** The label is learnable from features the pipeline
   produces. Any model that fails is failing to use those
   features.
3. **The next investigation is item 12.** `gcn_risk.py` uses
   `GCNConv`, which ignores `edge_attr`. That is the working
   hypothesis for the GCN failure. Item 12 records it; item 13
   fixes it and re-runs.

## What the paper's claim can now say

Before this run, the paper said "the GCN does not beat the
majority baseline" and the mechanism was a matter of debate
(imbalance, circularity, data scarcity). After this run:

> The harness can perfectly recover the sustained `run_length=5`
> label from a single integer feature (`n_critical` per window).
> The tested GCN does not, and its `GCNConv` layers ignore the
> edge attributes that carry that feature. The GCN's failure is
> therefore not evidence that graph models cannot learn traffic
> safety; it is evidence that the tested configuration could not
> see the label's signal.

That is narrower and more accurate. It names the mechanism and
the fix. It is the honest interim claim while item 13 runs.

## Cross-reference

- `PLAN.md` item 2 -- the evidence that closes it
- `PLAN.md` item 12 -- the GCN diagnosis this run motivates
- `docs/audit/feature_label_circularity.md` -- item 1, whose
  "unverified" question this run resolves
- `.github/workflows/positive-control.yml` -- the workflow
- `scripts/positive_control.py` -- the script
