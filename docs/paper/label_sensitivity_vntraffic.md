# Label sensitivity sweep — VNTraffic

**Question:** does any parameterization of the TTC-based window
label produce a task that a classical classifier can learn?
The declared-setup baseline comparison
(`baselines_vntraffic.md`) produced an 88/12 class split and
near-chance AUC for every model; this sweep tests whether that
is a label-definition artifact or the honest answer.

**Setup:** same 10 tracks, same 501 graphs, same blocked 5-fold
split as `baselines_vntraffic.md`. The only thing that changes
across configs is the label definition:

- TTC threshold grid: 0.5, 1.0, 1.5, 2.0, 2.5 seconds
- Distance threshold grid: 2.0, 3.0, 4.0, 5.0 metres
- Window rule: `any` (any frame has a TTC event), `center` (the
  middle frame only), `majority` (>50% of frames), `all` (every frame)

80 configs in total; 37 fell in the balanced
range [0.55, 0.80] and were evaluated with logistic regression
and random forest.

## Finding

**The sweep does not find a learnable configuration.** Under
every balanced label definition, no model meaningfully beats its
majority baseline on accuracy; AUC values above 0.5 appear but
with per-fold standard deviations of 0.13–0.29, so none is
statistically distinguishable from chance at n=5 folds.

### The balanced configs are the cleanest control

Configs where the class split is closest to 50/50 are the most
honest test — the majority-class shortcut provides no free
accuracy:

| config | pos | neg | majority | logreg (acc / auc) | rf (acc / auc) |
|---|---|---|---|---|---|
| (1.5, 3.0, center) | 125 | 124 | 0.502 | - | - |
| (1.5, 4.0, majority) | 124 | 125 | 0.502 | - | - |
| (1.5, 5.0, majority) | 135 | 114 | 0.542 | - | - |
| (2.0, 3.0, majority) | 133 | 116 | 0.534 | - | - |

At the essentially-balanced `(1.5s, 3.0m, center)` config
(pos=125, neg=124, maj=0.502), logistic regression achieves
accuracy 0.533 and random forest 0.583 — both at chance.
**Under a genuinely balanced task, the current features do not
separate positives from negatives.**

### The apparent AUC peak is not at the balanced configs

The strongest AUC in the sweep is **rf auc=0.750 ± 0.158** at `(2.5, 4.0, majority)`, where the majority rate is 0.659 — not balanced. If the features carried a
physical TTC signal, the model should perform best where the
class prior provides the least help, not where it provides the
most. The opposite pattern is consistent with the classifier
exploiting class-prior structure in the window statistics
rather than detecting the TTC-critical event itself.

## Implications

The negative result from `baselines_vntraffic.md` is not a
label-definition artifact. Across the tested grid of 80
parameterizations, no config produces a task where a classical
classifier cleanly separates positive from negative windows.

Two readings are consistent with the data:

1. **The negative result is real.** On this clip (10 actors, 501
   frames, VNTraffic), the TTC-critical event is either not
   learnable from the window-summary features at this sample size,
   or the features do not carry the discriminative information.
2. **The dataset is too small.** With 249 windows at stride 2 and a
   3-frame overlap, effective independent sample size is low. The
   per-fold AUC standard deviations in every row reflect this.

Disambiguating (1) from (2) requires more data or richer
features, not further tuning of the label definition.

## Next steps

- **Widen the track set.** The sweep used the 10 longest of 85
  tracks. Running on all 85 multiplies the number of actor pairs
  by ~72× and may reveal signal obscured by the top-10 filter.
- **Richer features.** `GraphFeatureExtractor` summarizes each
  window with ~50 aggregate statistics; the graph structure is
  discarded. A model reading edge features directly, or the GCN,
  may find structure the summary statistics do not carry.
- **More data.** Annotate additional clips or run on NSC footage.
  The decisive test is whether the finding holds at a dataset size
  with a reasonable number of negatives per fold.

## Full grid

See `label_sensitivity_vntraffic.json` for all 80 configs
including their label distributions and (for balanced ones)
model metrics.

