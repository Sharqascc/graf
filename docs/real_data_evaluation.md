# Real-data evaluation: VNTraffic

## Dataset

[VNTraffic](https://zenodo.org/records/18195750) from the Zenodo
"Vehicle Tracking" record.

- 1920x1080, 30 fps, 501 frames (~16.7 s)
- 85 annotated tracks with frame-level bounding boxes (MOT format)
- Hanoi intersection, motorbike-heavy mixed traffic

## Reproduce

```bash
# 1. fetch (~40 MB, one-time)
python scripts/fetch_vntraffic.py --data-root data/external/vntraffic

# 2. convert to GRAF inputs
python scripts/prepare_vntraffic.py --dataset-root data/external/vntraffic

# 3. build per-frame graphs (top-10 longest tracks recommended; see below)
python scripts/build_graphs.py \
    --tracks data/raw/vntraffic_tracks.jsonl \
    --output_dir data/processed/graphs/vntraffic \
    --homography_config data/raw/vntraffic_homography.yaml \
    --radius 10.0

# 4. per-frame SSM values
python scripts/compute_ssm.py \
    --traj-path data/interim/trajectories/vntraffic.json \
    --outdir data/processed/ssm_values/vntraffic \
    --video-id vntraffic --fps 30.0

# 5. mine SSM events
python scripts/mine_ssm_events.py \
    --values-path data/processed/ssm_values/vntraffic/ssm_values.jsonl \
    --outdir data/processed/ssm_events/vntraffic \
    --config-dir configs/ssm --min-duration-frames 3

## Reproducibility caveat (label definition)

The label definition recorded above (`ttc_threshold_seconds: 1.5`)
does **not** reproduce `num_positive: 86` from `real_data_metrics.json`.

A parameter sweep over distance thresholds (1.0-5.0 m), TTC
thresholds (0.5-2.0 s), source fps (23.98 and 30.0), track sets
(top-10 and all-85), and both window rules (any-frame and
center-frame) produces positive counts between 59 and 239. The
target 86 appears only as an isolated point at
`(distance=3.0, ttc=0.75, window=center)` with no plateau around it,
so it is a numerical coincidence, not a recovered definition.

The ad-hoc script that produced the original `0.547 / 0.611` numbers
was never committed and is no longer available. Those numbers are
retained here as a historical record, not as a baseline.

`scripts/evaluate_vntraffic.py` supports `--label-source ttc` with
configurable thresholds (`--ttc-threshold-seconds`,
`--ttc-distance-threshold`, `--ttc-closing-rate-threshold`) so that
an explicit, declared setup can be run and reported. Reproducing the
exact prior number is not achievable; running a declared setup and
reporting its honest result is.

## Declared-setup run (this repo, current main)

The original metrics above are unreproducible (see caveat). The
declared setup below is what this repo *can* run reproducibly:

| Parameter | Value |
|---|---|
| label source | ttc (window positive if any frame contains a TTC event) |
| ttc threshold | 1.5 s |
| ttc distance threshold | 3.0 m |
| ttc closing-rate threshold | 0.5 m/s |
| tracks | top-10 longest of 85 |
| fps | 30.0 (clip metadata) |
| window / stride | 5 / 2 |
| CV folds / epochs | 5 / 25 |
| device | cpu |

### Label distribution

249 windows: **220 positive** / **29 negative**, majority baseline **0.884**.

### Results

| metric | blocked CV | random CV |
|---|---|---|
| mean accuracy | 0.882 ± 0.068 | 0.884 ± 0.048 |
| mean F1 | 0.936 | 0.938 |
| mean AUC | 0.664 | 0.556 |
| pooled accuracy | 0.884 | 0.884 |
| pooled AUC | 0.390 | 0.492 |
| p vs majority | 0.5493 | 0.5493 |
| beats majority at p<0.05 | False | False |
| leakage per fold (val windows sharing frames with train) | [2, 4, 4, 4, 2] | [50, 50, 50, 50, 49] |

### Reading

1. **Neither split beats majority.** Pooled accuracy equals the
   majority baseline exactly; p = 0.55 in both cases.
2. **Leakage is not the explanation.** Blocked and random produce
   nearly identical mean accuracy (0.882 vs 0.884). The blocked
   split leaks only 2-4 windows per fold (fold-boundary effects),
   while the random split leaks 49-50 out of 50 val windows per
   fold. If leakage were inflating the model, the random split
   would be materially better — it is not.
3. **The declared setup is not the same task as the original run.**
   The original metrics record 86 positive / 163 negative (majority
   0.655). The declared TTC setup produces 220 positive / 29
   negative (majority 0.884). An 88/12 split with 29 negatives is
   nearly degenerate for binary classification; a model can get 88%
   accuracy by predicting positive for every window. AUC estimates
   on 29 negatives are correspondingly noisy.

### What would move this forward

The original label rule is unknown, but the shape of the label
distribution (65/35) suggests it produced many more negatives than
the current `ttc <= 1.5 in any frame` rule. Candidate explanations:
a stricter aggregation over the window (e.g. minimum TTC across
all frames, not any), a different definition of "event", or a
different track-selection rule. Without the original script this
is speculation; the current declared setup is what this repo
actually runs, and its honest result is the table above.

The leakage question, however, is now settled: on real sliding-
window data with the declared setup, blocking the CV split does
not change the result, so the original negative result is not a
leakage artifact.
