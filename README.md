# GRAF

**Graph-based surrogate safety analysis pipeline**

GRAF is a research pipeline for building graph representations of traffic interactions from video/detection/tracking data and applying graph-based models for surrogate safety analysis.

[![CI](https://github.com/Sharqascc/graf/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/graf/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project status

**Working end-to-end on synthetic data; not yet validated on real annotated video.**

| Area | Status |
|---|---|
| Library code (`src/graf/`) | Complete; 290 unit tests pass |
| Synthetic end-to-end test | `tests/test_end_to_end_synthetic.py` (~30 s on CPU) |
| Training API | `from graf.training import run_cross_validation` |
| Real-video pipeline | Runs, but no committed data or ground truth |
| Tracking quality metrics (MOTA / IDF1 / ID switches) | Not implemented |
| Reproducible real-video accuracy | None - see `docs/experiment_results.md` |

The 87.5% cross-validation accuracy in `docs/experiment_results.md` was produced
from local data that is not committed. See the reproducibility note there for
how to run a self-contained smoke test instead.

## Features

- **Detection & Tracking** – interfaces for YOLOv8, RT‑DETR, ByteTrack, BotSORT
- **Homography Calibration** – image→world coordinate transforms and ROI handling
- **Graph Construction** – spatial interaction graphs with class‑specific radii and kinematic edge features
- **Spatio‑Temporal Graphs** – rolling window graphs that connect actors across frames
- **Surrogate Safety Measures** – TTC, PET, DRAC, and event mining
- **Graph Models** – GCN, ST‑GCN, graph transformers, and classic baselines
- **Evaluation** – binary classification metrics, calibration, robustness analysis
- **Reproducible Pipeline** – configuration files, logging, experiment tracking, and CI

## Installation

### CPU only

```bash
pip install -r requirements/base.txt -r requirements/dev.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
