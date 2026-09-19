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
