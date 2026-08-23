# GRAF

**Graph-based surrogate safety analysis pipeline**

GRAF is a research pipeline for building graph representations of traffic interactions from video/detection/tracking data and applying graph-based models for surrogate safety analysis.

[![CI](https://github.com/Sharqascc/graf/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/graf/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
