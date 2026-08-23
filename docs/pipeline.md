# Pipeline

## Overview

GRAF transforms traffic video into graph-based representations for surrogate safety analysis. The pipeline is modular and can be run end-to-end or stage-by-stage.

## Stages

1. **Raw data ingestion**
   Inputs: videos, site metadata, annotations.
   Outputs: organized raw data under `data/raw/`.

2. **Frame extraction**
   `scripts/extract_frames.py` decodes videos into frames.

3. **Detection**
   `scripts/run_detection.py` uses YOLOv8 or RT-DETR.
   Config: `configs/detection/*.yaml`.

4. **Tracking**
   `scripts/run_tracking.py` uses ByteTrack or BotSORT.
   Config: `configs/tracking/*.yaml`.

5. **Homography & world coordinates**
   `scripts/estimate_homography.py` calibrates camera.
   Modules: `graf.calibration.homography`, `world_coords`.

6. **Trajectory construction**
   `scripts/build_trajectories.py` creates smooth world-space trajectories.
   Modules: `graf.trajectories`.

7. **Surrogate safety measures**
   `scripts/compute_ssm.py` computes TTC, PET, DRAC.
   Modules: `graf.ssm`.

8. **Graph construction**
   `scripts/build_graphs.py` builds interaction graphs.
   Modules: `graf.graph.builders`, `features`, `temporal`.

9. **Temporal window dataset**
   `scripts/make_windows.py` creates rolling windows.
   Dataset: `graf.data.graph_dataset.SpatioTemporalWindowDataset`.

10. **Model training & evaluation**
    `scripts/train_model.py`, `scripts/evaluate_model.py`.
    Modules: `graf.models`, `graf.training`, `graf.evaluation`.

## Key directories
