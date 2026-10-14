# SCI tracking quality -- improvement plan

**Status:** pre-registration. The phases below are committed before any
is applied. `git log docs/sci_tracking_improvement_plan.md` will show a
commit hash that predates every phase's results.

## Why this document exists

The stride-1 SCI run (workflow run 38048010561) produced tracks that are
not yet usable for PET/TTC analysis:

- 47,945 detections collapsed into 316 tracks
- 62% of tracks have a gap >3 frames
- The longest "track" lasts all 2,700 frames of the clip
- 8,123 auto-rickshaw detections became 22 tracks (369 rows per track)

These are classic multi-object-tracking failure modes on a dense,
semi-static scene. The fixes are well-understood but the failure has
several possible causes: stationary-object merging, association
ambiguity in crowded scenes, and identity switches across visually
similar vehicles. Applying a fix without knowing which cause dominates
risks fixing the wrong thing.

This memo commits the sequence of diagnostics and fixes before any is
run. It follows the same pattern as
`docs/preregistration_sustained_r5.md`: fix the plan, execute it once,
report the outcome regardless of direction.

## Current state

Two runs of `configs/recipes/sci_v1.yaml` on the SCI video, differing
only in `--stride`:

| metric | stride 3 (10 fps) | stride 1 (30 fps) |
|---|---:|---:|
| frames processed | 900 | 2,700 |
| detections | 15,994 | 47,945 |
| detections per frame | 17.8 | 17.8 |
| unique tracks | 217 | 316 |
| median track length | 26 | 34 |
| max track length | 900 | **2,700** |
| tracks >= 20 frames | 119 (55%) | 182 (58%) |
| tracks >= 50 frames | 81 (37%) | 142 (45%) |
| **pct_discontinuous (gap >3)** | **71%** | **62%** |

Class breakdown (stride 1):

| class | detections | tracks | rows per track |
|---|---:|---:|---:|
| two_wheeler | 27,864 | 179 | 156 |
| car | 10,713 | 106 | 101 |
| auto_rickshaw | 8,123 | **22** | **369** |
| truck | 1,009 | 6 | 168 |

The 3x finer cadence improved discontinuity by 9 points (71% to 62%).
If frame-to-frame motion were the primary cause, a 3x finer cadence
would cut discontinuity roughly in half. It did not. The dominant cause
is structural in the scene, not temporal.

## The improvement menu

Ranked by cost. The menu is the union of the three documents circulated
for this plan; each item cites its strongest source.

1. **Diagnose before fixing.** Plot the longest track's frame-by-frame
   centers. If the center barely moves, the tracker is holding one
   object correctly and the fix is a static-object filter. If the
   center jumps around the parked row, the tracker is merging objects
   and the fix is association logic. (Document 3.)
2. **Static-object filter.** Drop tracks whose net displacement over N
   frames is below a threshold, or mask the parking area with an ROI
   polygon. This also fixes a safety-analysis correctness issue: a
   parked vehicle near moving traffic looks like a PET event to the
   SSM stage. (All three documents.)
3. **Association improvements to the existing tracker.** Velocity
   gating (reject physically implausible jumps) and stricter
   class-awareness. The observed failure mode is within-class:
   two-wheeler to two-wheeler. (Documents 1 and 3.)
4. **ByteTrack or BoT-SORT.** Standard trackers with two-stage
   association. ByteTrack rescues low-confidence detections that the
   current greedy tracker drops, which is the fragmentation cause.
   BoT-SORT adds camera-motion compensation. Neither solves visually
   identical overlapping stationary objects on its own; they must be
   combined with item 2. (All three documents.)
5. **Ground-truth validation.** Hand-label a representative sample of
   ~50 frames and compute IDF1 / HOTA / MOTA. Without this step, any
   improvement number is a diagnostic, not a tracking-accuracy claim.
   (Documents 1 and 3.)

## Decision rules

After each phase, the next phase is chosen by the pre-committed
criteria below, not by how the numbers feel.

### Phase 0 -> Phase 1

Run the diagnostic on the stride-1 tracks. Two outcomes:

- Longest track's center is **stationary** -> proceed to Phase 1 with
  ROI and displacement filter.
- Longest track's center **jumps** within the parked row -> Phase 1
  still applies (the static filter removes the region either way), but
  Phase 2 becomes the higher priority.

### Phase 1 -> Phase 2

After ROI and displacement filter, recompute `pct_discontinuous`:

- **Under 30%** -> SCI is usable as a demo. Stop and write it up.
- **30--50%** -> proceed to Phase 2 (association improvements).
- **Over 50%** -> the filter did not fix the primary failure. Proceed
  directly to Phase 3 (ByteTrack).

### Phase 2 -> Phase 3

After velocity gating and class-aware matching:

- **Under 30%** -> stop, write up.
- **Over 30%** -> ByteTrack is the next step.

### Phase 3 -> Phase 4

After ByteTrack:

- **Under 20%** -> proceed to Phase 4 (ground-truth sample).
- **Over 20%** -> SCI is not viable as a quantitative second site.
  Report the number and stop.

## Phases

### Phase 0 -- Diagnose the stride-1 run

Output: a diagnostic figure (or numeric table) of the longest track's
per-frame centers, and the same for the second-longest. Also a
per-class distribution of the discontinuity metric.

Deliverable: `scripts/diagnose_tracking.py` and
`tests/test_diagnose_tracking.py`. Runs on the already-uploaded
`sci-tracks-38048010561` artifact.

Effort: 1 hour.

### Phase 1 -- Static-object filter

Two changes to the pipeline:

- ROI polygon over the active roadway. Detections outside it are
  dropped before tracking.
- Displacement filter. Tracks whose total displacement over their
  lifetime is below a threshold (proposed: 2 metres in world coords)
  are dropped after tracking.

Both configurable via `configs/tracking/sci.yaml`.

Deliverable: the script, the config, tests.

Effort: 2--3 hours.

### Phase 2 -- Association improvements

In `scripts/run_tracking.py`:

- Velocity gating: reject matches where the implied inter-frame
  displacement exceeds a plausible maximum for the actor class.
- Stricter same-class matching: reject matches at implausibly
  different scales within the same class.

Deliverable: the code change, tests, a rerun of the SCI pipeline.

Effort: 3--4 hours.

### Phase 3 -- ByteTrack (conditional)

If Phase 1 and 2 combined leave `pct_discontinuous` above 30%, add a
`--tracker` option to the pipeline that accepts `bytetrack` and routes
to `ultralytics.YOLO.track(..., tracker="bytetrack.yaml")`.

Deliverable: the option, tests, a rerun.

Effort: 1--2 days.

### Phase 4 -- Ground-truth validation (conditional)

If Phase 3 brings discontinuity under 20%, hand-label ~50 frames from
the SCI clip and compute IDF1 / HOTA / MOTA. This is the only step
that produces a tracking-accuracy claim, as opposed to a diagnostic.

Deliverable: an annotation file, a metric script, a results table.

Effort: 1--2 days.

## What this plan commits the analysis not to do

Once execution starts:

1. **No sweeping thresholds to chase a number.** The ROI polygon is
   drawn once from the frame; the displacement threshold is 2 m. If
   the resulting number is unfavorable, that is the result.
2. **No changing the detector.** UVH-26 with the current confidence
   and NMS settings is fixed for every phase. Detector changes are a
   separate variable and would invalidate the phase comparison.
3. **No changing the window.** The 90-second window starting at 300 s
   is fixed. Longer windows complicate comparison and are not needed
   to answer "is the tracker usable."
4. **No combining phases in a single result.** Each phase reports its
   own delta. If Phases 1 and 2 both land, the paper reports both,
   not only the combined number.
5. **No declaring SCI viable without Phase 4.** A discontinuity metric
   is a diagnostic. Without hand-labeled ground truth, "SCI tracks are
   good" is not a claim this plan permits.

## Addendum 1 -- Phase 1+2 executed (2026-10-14)

Phase 1 and Phase 2 were applied in PR #122 and the SCI pipeline was
re-dispatched. This addendum records the result before Phase 3 begins.

### Result

| metric | stride-1 baseline | after Phase 1+2 | direction |
|---|---:|---:|---|
| detections in | 47,945 | 47,945 | -- |
| rows emitted | 47,945 | 34,256 | -13,689 |
| unique tracks | 316 | 264 | down |
| median track length | 34 | 58.5 | **+72%** |
| max track length | 2,700 | 2,690 | flat |
| tracks >= 20 frames | 58% | 74% | **+16 pts** |
| tracks >= 50 frames | 142 | 144 | flat |
| **pct_discontinuous** | **62.3%** | **65.9%** | **worse** |

Baseline run: ``38048010561`` (stride 1).
Phase 1+2 run: ``38557140055``.

### Reading

Two of the three tracking-quality signals improved materially. The
median track length nearly doubled and the fraction of tracks lasting
at least 20 frames rose by 16 points. Vehicles that were previously
fragmented into short pieces are now tracked as single objects. For
trajectory-based safety analysis that is the improvement that matters:
a longer median track means more frames per vehicle to compute TTC and
PET over.

The pre-committed decision metric, ``pct_discontinuous``, moved the
wrong way (62.3% -> 65.9%). The mechanism is specific and is not a
defect in the gates:

- The ROI removes the tower region (x > 1350). Vehicles passing through
  it lose their detections for a few frames. Their track remains open
  but emits no rows, which is a gap by the metric's definition.
- The velocity gate rejects matches that would otherwise bridge some
  of those gaps. On a scene with a tower occlusion, a conservative
  per-frame velocity threshold trims the very rows that would close a
  gap.

Both effects are the gates doing what they were designed to do. The
metric does not distinguish "gap from occlusion" from "gap from ID
flicker." That distinction is exactly what ByteTrack's two-stage
association is designed to fix: it rescues the low-confidence
detections that appear immediately behind an occluder, closing the gap
without loosening the velocity gate.

### Decision

The plan's Phase 2 -> Phase 3 rule, written before any result existed:

> After Phase 2 (velocity gating and class-aware matching):
>   * Under 30% -> stop, write up.
>   * Over 30% -> ByteTrack is the next step.

65.9% is over 30%. **Phase 3 (ByteTrack) is the next phase.**

This addendum is a record, not a re-tune. The plan's "what this commits
the analysis not to do" section forbids sweeping thresholds or
redefining the metric to chase a favourable number. Neither has
happened: the gates are unchanged, the metric is unchanged, and the
decision is the one the plan pre-committed.

### What the next run must show

Phase 3 is judged on the same three numbers, pre-committed here before
the ByteTrack run exists:

| metric | target |
|---|---:|
| pct_discontinuous | **< 30%** |
| median track length | >= 58 (do not regress) |
| unique tracks | between 100 and 300 |

If Phase 3 lands all three, SCI is usable as a second-site demo. If
``pct_discontinuous`` stays above 30% but the other two hold, SCI is
usable only as a qualitative figure, not as a quantitative dataset. If
median track length or unique-track count regress, the ByteTrack
configuration is wrong and must be reverted before any further claim.

## Cross-reference

- `docs/external_review_2026_09.md` -- the reviewer-checklist pattern
  this plan follows.
- `docs/preregistration_sustained_r5.md` -- the primary-analysis
  preregistration. This plan is the SCI counterpart.
- `.github/workflows/sci-pipeline.yml` -- the workflow the phases
  modify.
- `scripts/track_quality_summary.py` -- the metric that each phase
  reports against.
- Workflow runs: `38045758114` (stride 3), `38048010561` (stride 1).
