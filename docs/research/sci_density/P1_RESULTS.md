# P1 results: window-level TTC labels on SCI

**Status:** first results, 2026-11-10. One site, no independent ground
truth. Claim ceiling stated in the last section.

## Method chain

Data: `sci.mp4` (release asset `sci-v1`, 1.4 GB), processed by
`.github/workflows/sci-pipeline.yml`.

| stage | script | trained by us? |
|---|---|---|
| frames | `extract_frames.py`, 90 s at 300 s, stride 1, 2700 frames | no |
| detections | `run_detection.py`, UVH-26-MV-YOLOv11-S | no, pretrained |
| tracks | `run_tracking.py`, `configs/tracking/sci.yaml` | no |

Output: `tracks.jsonl`. 34,256 rows, 264 tracks after cleaning.

**No classifier was trained for the finding.** The analysis is:

1. Per-frame per-pair TTC with the B.1 DCPA gate: a pair is critical
   when `dist < 3.0 m`, `closing_rate > 0.5 m/s`,
   `0 < TTC <= 1.5 s`, and `DCPA < 1.5 m`.
2. Eight window aggregation rules: `any`, `sustained_r5`,
   `sustained_r30`, `min_ttc_0.5`, `min_ttc_1.0`, `pair_frac_0.05`,
   `pair_frac_0.10`, `persistent_r3`.
3. Permutation null test (20 trials, seed 42): each track's
   `frame_idx` is independently circular-shifted by a random offset.
   Track durations and per-frame motion preserved; pair-specific
   spatial co-occurrence destroyed.
4. Secondary: blocked-CV random forest on 8 leak-free scalar features.

Scale auto-estimated at 29.89 px/m. TTC is scale-invariant.

## Results

### 1. Saturation (1275 windows, 5 s, stride 2)

| label | positives | rate |
|---|---:|---:|
| any_1.5 | 1275 | 1.000 |
| sustained_r5 | 1275 | 1.000 |
| sustained_r30 | 137 | 0.107 |
| min_ttc_0.5 | 1275 | 1.000 |
| min_ttc_1.0 | 1275 | 1.000 |
| pair_frac_0.05 | 1275 | 1.000 |
| pair_frac_0.10 | 1275 | 1.000 |
| persistent_r3 | 1275 | 1.000 |

### 2. Permutation null (20 trials)

| label | real | null mean | null std | delta |
|---|---:|---:|---:|---:|
| any_1.5 | 1.000 | 0.999 | 0.003 | +0.001 |
| sustained_r5 | 1.000 | 0.986 | 0.025 | +0.014 |
| sustained_r30 | 0.107 | 0.451 | 0.089 | -0.344 |
| min_ttc_0.5 | 1.000 | 0.998 | 0.006 | +0.002 |
| min_ttc_1.0 | 1.000 | 0.999 | 0.005 | +0.001 |
| pair_frac_0.05 | 1.000 | 0.999 | 0.003 | +0.001 |
| pair_frac_0.10 | 1.000 | 0.999 | 0.003 | +0.001 |
| persistent_r3 | 1.000 | 0.989 | 0.021 | +0.011 |

### 3. Frame-level persistence

| scene | % critical frames | longest run | r30 rate | r5 rate |
|---|---:|---:|---:|---:|
| real | 55.9 | 59 | 0.107 | 1.000 |
| null (mean of 5) | 59.2 | 103.0 | 0.451 | 0.989 |

### 4. Density (secondary)

Blocked-CV AUC of density-only predictor on `sustained_r30`: **0.487**
(at chance).

### 5. Pair-level census (real scene)

314 distinct critical pairs. Median per-pair critical duration 5
frames. Longest single pair: 27 frames (1.0% of clip). No pair
persists.

## Interpretation

**Finding 1 (robust).** Seven of eight label variants are
indistinguishable from a null that destroys every real spatial
interaction. The saturated labels are opportunity counters, not
interaction detectors. This holds regardless of tracking quality
because null and real rates are both at 1.000.

**Finding 2 (provisional).** `sustained_r30` shows real = 0.107 vs
null = 0.451, a gap of 4 standard deviations. Two competing
explanations, not distinguished by this analysis:

- (a) Real drivers resolve conflicts; random tracks cannot, so null
  sustained runs persist.
- (b) The tracker fragments tracks when real vehicles maneuver
  (`pct_discontinuous = 65.9%` in
  `docs/sci_tracking_improvement_plan.md`). Null tracks are smooth
  circular shifts that cannot fragment.

Without the Phase 3 ByteTrack improvement (`pct_discontinuous < 30%`),
we cannot say whether the gap reflects avoidance behavior or tracking
failure. **Do not cite Finding 2 as avoidance until that test is
done.**

## What this means for the paper

Both findings point to the same limitation: window-level aggregation
of per-pair SSM events flattens per-pair information into scene-level
statistics. The saturated labels measure opportunity; the strict label
measures scene dynamics (possibly including tracker artifacts).
Neither measures per-pair safety risk.

That is the paper's claim. It is not "TTC is wrong" and it is not "the
field is broken." It is "the aggregation rule, not the physics,
determines what the label measures."

## Scope and claim ceiling

- One site. SCI is a case study, not a field-wide confirmation.
- No independent ground truth. No human near-miss labels.
- Tracking quality confound on Finding 2.
- Claim ceiling: on one heterogeneous-traffic site, seven of eight
  window-level TTC labels are statistically indistinguishable from a
  null that destroys real spatial interactions.

## Reproduce

python scripts/sci_density_analysis.py with flags:
--tracks tracks.jsonl
--output_dir docs/research/sci_density
--win_frames 150 --stride_frames 2
--permutation-trials 20

Source artifact: SCI pipeline run 42638434164, artifact
sci-tracks-42638434164 (id 13685821606), regenerated 2026-11-09.
