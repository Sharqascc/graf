# Feature-label circularity audit (item 1)

**Verdict: TTC-derived information IS present in the input features,**
**via aggregates over `edge_attr`.** The label is a thresholded
function of TTC. Together these make the primary comparison
near-circular: no model in the current harness is doing prediction —
all are partly reconstructing the labeling rule.

## What the label is

A 5-frame window is `critical` (label = 1) if any frame inside it
contains a pair with `TTC <= 1.5s`. See `scripts/compare_baselines.py`
and `scripts/evaluate_vntraffic.py`. The threshold is 1.5 s and the
distance gate is 3.0 m, both fixed by pre-registration.

## What the input features are

The 42-column vector is produced by
`src/graf/models/baselines.py::GraphFeatureExtractor._graph_to_vector`.
The names and order are in `_FEATURE_NAMES` (baselines.py:60-103):

| index | name | source |
|---:|---|---|
| 0-5 | num_nodes, num_edges, density, *_dim | graph shape |
| 6-14 | x_mean/std/min/max, x_row_l2_*, x_nonzero_frac | `graph.x` aggregates |
| 15-23 | edge_attr_mean/std/min/max, edge_row_l2_*, edge_attr_nonzero_frac | `graph.edge_attr` aggregates |
| 24-27 | pos_mean/std/min/max | `graph.pos` aggregates |
| 28-31 | y_mean/std/min/max | `graph.y` aggregates |
| 32-41 | frame_id, video_id, track_id, node_frame_index, actor_class_index (mean, unique) | metadata |

`_safe_stats(arr)` (baselines.py:40-52) returns `[mean, std, min, max]`
of `arr.flatten()`. Applying it to `edge_attr` **flattens the 15 edge
feature columns into one bag** before computing any statistic.

## Where TTC lives

`edge_attr` has 15 columns, documented in
`src/graf/graph/builders.py::GraphBuilder` docstring:

```
0  dx         5  relative_speed    10 rel_heading_sin
1  dy         6  closing_speed     11 ttc
2  distance   7  bearing_sin       12 same_class
3  dvx        8  bearing_cos       13 size_src
4  dvy        9  rel_heading_cos   14 size_dst
```

**TTC is column index 11.** `closing_speed` (6), `relative_speed`
(5), and `distance` (2) are also TTC-adjacent: TTC is
distance / closing_speed.

## Which of the 42 columns carry TTC

| index | name | mechanism |
|---:|---|---|
| 15 | edge_attr_mean | mean over 15-col flatten, TTC is one of 15 |
| 16 | edge_attr_std | same |
| 17 | edge_attr_min | same |
| 18 | edge_attr_max | same |
| 19 | edge_row_l2_mean | L2 of each row includes ttc squared |
| 20 | edge_row_l2_std | same |
| 21 | edge_row_l2_min | same |
| 22 | edge_row_l2_max | same |
| 23 | edge_attr_nonzero_frac | TTC entries are nonzero when TTC exists |

**9 of 42 columns (21%) carry TTC-derived information.**

## The winning single-feature baseline

`edge_attr_nonzero_frac` (column 23) is the trivial cue that beats
the random forest on every reported metric. It is the fraction of
nonzero entries across `edge_attr`. The TTC column is nonzero
exactly when a pair has a finite TTC. So `edge_attr_nonzero_frac`
is mechanically correlated with whether some pair has a finite
TTC, which is the label's own trigger condition.

This is the mechanism the reviewers named. The scalar is not a
physical law of traffic safety; it is a one-number proxy for the
label itself.

## The y columns (28-31)

`y_mean`, `y_std`, `y_min`, `y_max` aggregate `data.y`.

`builders.py` sets `y` to one element or empty per-graph. The
graph_dataset docstring says labels live at window level, not
frame level, so `data.y` should be empty when
`compare_baselines.py` recomputes labels. If empty, these four
columns are always 0.0 and carry no information.

**Status: unverified.** Item 2's positive control should confirm
they are zero on the current graph-build config. If
`build_graphs.py` ever starts writing the frame label into
`data.y`, these four columns become direct target leakage.

## What this changes about the paper's claim

The current sentence in `conclusion.md` says a single scalar
feature matches or beats the trained models. That is factually
true about the harness. But it reads as a claim about traffic
safety. With this audit, it becomes a claim about a pipeline whose
label is a thresholded function of an input feature.

The honest claim is: **under a TTC-defined window label, with
TTC-inclusive edge features, no model — including a graph
convolutional network — beats a single-feature scalar that is
mechanically correlated with the label's own trigger condition.**

That is narrower, still defensible, and it names the mechanism.

## What clean would require

1. **Remove TTC-derived edge features.** Drop columns 2, 5, 6, 11
   from `edge_attr` before aggregation. That removes `distance`,
   `relative_speed`, `closing_speed`, and `ttc`.
2. **Re-run `compare_baselines.py`.** If RF AUC collapses toward
   0.5, TTC was doing the work. If it holds, some other signal
   exists and is worth characterizing.
3. **Run the ablations in `PLAN.md` item 4:** node features only;
   edge features with TTC removed; edge features with only
   dx, dy, same_class, size_src, size_dst.
4. **Then move to the forecasting formulation** (`PLAN.md` item 6):
   features at window t, label at window t+k, TTC-derived
   features excluded. That is the non-circular version of the task.

## Test that enforces this discovery

`tests/test_audit_circularity.py` checks that the audit exists,
names all 9 edge columns, cites the edge indices of TTC,
prescribes the clean path, and catches a future refactor that
silently adds a column literally named `ttc` to `_FEATURE_NAMES`.

## Cross-reference

- `PLAN.md` item 1 — this document is the evidence that closes it
- `PLAN.md` item 2 — positive control (tree on min-TTC approximately 1.0)
- `PLAN.md` item 3 — density confound
- `PLAN.md` item 4 — metric suite
- `docs/audit/consensus_improvements.md` — the three reviews that
  named this as the biggest risk
