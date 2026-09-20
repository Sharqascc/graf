# Baseline comparison — VNTraffic, all-81 tracks (degenerate)

**Companion to:** `baselines_vntraffic.md` (top-10 tracks).

**Result: the label set is fully degenerate at this track count.**
Of 249 windows, **249 are positive** under the declared label
(TTC ≤ 1.5 s within 3.0 m in any frame of the window). No model
can be trained or evaluated. This is a finding, not a failure.

## Setup

- **Tracks:** all 81 tracks with ≥20 frames of the 85 in the clip.
  The 4 excluded are fragments of 1, 3, 6, and 12 frames.
- **Label source:** TTC (window positive if any frame has
  0 < TTC ≤ 1.5 s within 3.0 m)
- **Windows:** window_size=5, stride=2 → 249 windows
- **Class balance:** 249 positive / 0 negative (majority 1.000)

## Reading

With 81 actors in frame, some pair of tracks is inside 3 m of each
other with a closing rate that produces TTC ≤ 1.5 s in essentially
every 5-frame window. The label rule "any frame contains a TTC
event" therefore collapses: it cannot distinguish "a conflict
occurred" from "traffic as usual" when many actors share the
scene.

## Comparison to top-10

| track set | windows | positives | negatives | majority |
|---|---|---|---|---|
| top-10 (baselines_vntraffic.md) | 249 | 220 | 29 | 0.884 |
| all-81 (this file) | 249 | 249 | 0 | 1.000 |

Widening the track set does not improve the classification task;
it makes it strictly more degenerate. More actors means more
opportunities for a TTC event to occur by chance, and the "any
frame" rule aggregates those opportunities into a positive label
for every window.

## Implications

This is a stronger negative result than the top-10 run showed, and
it is more pointed. The standard SSM labeling approach — "the
window is critical if a TTC event occurs" — is not scale-robust.
It works as a per-pair, per-frame measure of conflict severity but
fails as a window-level classification target because it
saturates when there are many actors.

Two directions follow, and they are stated in the paper's
Discussion as the honest limitations of this work:

1. **Label design is the bottleneck, not the model.** A window-level
   target that is robust to actor count would need to normalize by
   the number of pairs in the window, or select only the most
   critical pair, or require sustained conflict rather than any
   single frame.
2. **Wider track sets are the wrong direction.** The intuition
   "more actors = more data = better training" fails here because
   the label collapses. The next lever is not more tracks but a
   more discriminative label.

## Data

See `baselines_vntraffic_all81.json` for the raw label counts.
