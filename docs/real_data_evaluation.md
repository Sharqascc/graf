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

# 3. build per-frame graphs
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
