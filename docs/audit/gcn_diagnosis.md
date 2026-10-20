# GCN failure diagnosis (item 12)

**Verdict: the tested GCN was never given a fair test.** Its
convolutional layers are `GCNConv`, which has no `edge_dim`
parameter and structurally cannot accept edge features. The
15-dimensional `edge_attr` that carries `ttc`, `distance`,
`relative_speed`, and `closing_speed` never reaches the model.
The model's AUC of 0.590 is the ceiling of what the 19 node
features can produce, not evidence that graph models cannot
learn this label.

## The code

`src/graf/models/gcn_risk.py:96-98`:

     self.convs = nn.ModuleList()
     self.convs.append(GCNConv(in_channels, hidden_channels))
     for _ in range(num_layers - 1):
         self.convs.append(GCNConv(hidden_channels, hidden_channels))

`src/graf/models/gcn_risk.py:106`:

     x, edge_index = data.x, data.edge_index

`data.edge_attr` is never read. `GCNConv.forward(x, edge_index,
edge_weight=None)` accepts only a scalar per-edge weight. The
15-dim tensor in `edge_attr` is silently ignored.

## What those 15 dims contain

From `GraphBuilder`'s docstring in `builders.py`:

     index  feature
     0      dx
     1      dy
     2      distance
     3      dvx
     4      dvy
     5      relative_speed
     6      closing_speed
     7      bearing_sin
     8      bearing_cos
     9      rel_heading_sin
     10     rel_heading_cos
     11     ttc
     12     same_class
     13     size_src
     14     size_dst

**TTC is column 11.** `distance` (2), `relative_speed` (5), and
`closing_speed` (6) are TTC-adjacent: TTC is `distance /
closing_speed`. All four are ignored by the current model.

## The positive control proves this matters

`docs/audit/positive_control.md` (item 2) shows:

| feature | tree accuracy |
|---|---:|
| `n_critical` per window | 1.0 |

A depth-1 decision tree on a single integer recovers the harness
label with 100% accuracy. The label is learnable from features
the pipeline produces. The GCN cannot learn it because the
features that carry the signal are not connected to the model.

## Why the existing tests missed it

`tests/test_gcn_risk.py` has 6 tests:

| test | asserts |
|---|---|
| `test_has_torch_geometric` | library imports |
| `test_build_model_returns_real_model` | class name |
| `test_build_model_forward_single_graph` | output shape `(1,)` |
| `test_build_model_forward_batched_graphs` | output shape `(2,)` |
| `test_invalid_num_layers_raises` | error on num_layers<2 |
| `test_invalid_pool_raises` | error on unknown pool |

Deleting `edge_attr` from every graph would leave all 6 green.
Coverage was 80% but behavior coverage of the actual failure
mode was 0. This is the class of bug that coverage cannot catch:
the lines execute, but the properties they should implement are
not asserted.

## The 8 behavioral tests that should exist

Added in `tests/test_gcn_behavior.py` by this item. Two are
`xfail(strict=True)`:

| test | passes on current code? | catches |
|---|---|---|
| `test_output_depends_on_edge_attr` | **NO (xfail)** | `GCNConv` bug — the smoking gun |
| `test_model_accepts_edge_dim_kwarg` | **NO (xfail)** | missing `edge_dim` interface |
| `test_output_shape_single_graph` | yes | regression guard |
| `test_output_shape_batched_graph` | yes | regression guard |
| `test_determinism_same_seed` | yes | determinism |
| `test_gradients_reach_every_parameter` | yes | dead gradients |
| `test_pool_variants_produce_different_outputs` | yes | pooling wiring |
| `test_model_state_dict_roundtrip` | yes | checkpoint validity |

**Why `xfail(strict=True)` and not skip:** the test fails today
because the model genuinely cannot see edge_attr. Marking it
xfail lets the PR merge without breaking CI. Marking it
**strict** means the moment item 13 fixes the model and this
test starts passing, CI fails with "unexpectedly passed" —
forcing the developer to remove the marker and confirm the fix
is real. The test cannot silently be forgotten.

## What item 13 will do

1. Replace `GCNConv` with an edge-aware layer (`GINEConv`,
   `TransformerConv`, `NNConv`, or `GATv2Conv`) that accepts
   `edge_dim=15`.
2. Pass `data.edge_attr` into each conv in `forward`.
3. Change the default pool from `mean` to `max` (mean dilutes a
   single critical edge across ~20 vehicles; max preserves
   "some pair is critical").
4. Remove the two `xfail(strict=True)` markers when the tests
   pass.
5. Re-run `scripts/compare_baselines.py`. Record the new GCN
   AUC in `docs/paper/results.md`.

## What this changes about the paper's claim

Before item 12: *"the GCN adds nothing over a scalar baseline."*
That sentence was true of the harness's output but wrong about
its cause — the model never saw the signal.

After item 13's re-run, one of two outcomes:

- **New GCN AUC >= 0.757** (matches or beats RF on the same
  features): the previous result was a bug. Rewrite results.
- **New GCN AUC < 0.757**: the claim survives with a different
  mechanism — *even with edge features, message passing does
  not beat a single aggregated scalar on a near-circular label.*
  Still honest, still publishable.

Either way, the interim claim while item 13 is pending must
acknowledge the model-architecture limitation.

## Cross-reference

- `PLAN.md` item 12 — this item
- `PLAN.md` item 13 — the fix and re-run
- `docs/audit/feature_label_circularity.md` — item 1, the
  circularity finding this diagnosis is compatible with
- `docs/audit/positive_control.md` — item 2, proving the label
  is learnable
- `src/graf/models/gcn_risk.py` — the code under diagnosis
- `tests/test_gcn_risk.py` — the existing tests, and what they
  missed
- `tests/test_gcn_behavior.py` — the tests this item adds
