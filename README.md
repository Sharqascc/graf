# GRAF

**Graph-based surrogate safety analysis pipeline**

GRAF is a research pipeline for building graph representations of traffic interactions from video/detection/tracking data and applying graph-based models for surrogate safety analysis.

[![CI](https://github.com/Sharqascc/graf/actions/workflows/ci.yml/badge.svg)](https://github.com/Sharqascc/graf/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Quickstart

```bash
git clone https://github.com/Sharqascc/graf.git
cd graf
pip install -e ".[dev]"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric ultralytics

# 1. check that your environment is ready
graf doctor

# 2. see all available commands
graf --help

# 3. reproduce the paper's primary result
graf reproduce configs/recipes/paper_v1.yaml
```

If `graf doctor` reports missing detector weights, fetch them:

```bash
graf fetch-detector --model YOLOv11-S
```

The `graf` command is the entry point for the pipeline. See [CLI](#cli)
for the full command list, and [Recipes](#recipes) for how the paper's
results are reproduced.

## Project status

**Working end-to-end on synthetic data; not yet validated on real annotated video.**

| Area | Status |
|---|---|
| Library code (`src/graf/`) | Complete; ~600 unit and property tests |
| CLI (`graf`) | Entry point; 7 subcommands - see [CLI](#cli) |
| Recipes | `configs/recipes/paper_v1.yaml` reproduces the paper's primary result |
| Synthetic end-to-end test | `tests/test_end_to_end_synthetic.py` (~30 s on CPU) |
| Detection | UVH-26 YOLOv11-S (India-specific); results in `docs/paper/detection_*.md` |
| Tracking quality metrics (MOTA / IDF1 / ID switches) | Not implemented |
| Reproducible real-video accuracy | Run `graf reproduce configs/recipes/paper_v1.yaml` |

The 87.5% cross-validation accuracy in `docs/experiment_results.md` was produced
from local data that is not committed. See the reproducibility note there for
how to run a self-contained smoke test instead.

## Known issues and caveats

Two results in `docs/paper/` predate the cross-validation boundary-leak fix
in [PR #39](https://github.com/Sharqascc/graf/pull/39) (commit `79a937e`):

- **RF AUC 0.782** on the sustained `run_length=5` label — the *pre-purge*
  number. The post-purge number requires regenerating the input tracks.
- **"5.7 SEs above chance"** in
  [`docs/paper/label_strategies_vntraffic.md`](docs/paper/label_strategies_vntraffic.md)
  — **retracted**. The standard-error calculation assumes independent folds,
  which boundary-leaked folds violate.

The full audit and its resolution log is in
[`docs/external_review_2026_09.md`](docs/external_review_2026_09.md).

### Data availability

The real-video results depend on three gitignored, generated artifacts:

- `data/raw/vntraffic_homography.yaml`
- `data/interim/vntraffic_tracks_all85_min20.jsonl`
- `data/processed/graphs/vntraffic_all85/`

These were generated on an ephemeral Colab VM that has since been recycled.
The pipeline is unchanged; the numbers can be reproduced once the source
videos are re-ingested.

### Trivial-cue baseline

The paper's central comparison — RF on all 42 features vs a single scalar
(`edge_attr_nonzero_frac`) — is reproducible from the CLI as of
[PR #40](https://github.com/Sharqascc/graf/pull/40):

```bash
# The paper's primary comparison is the `compare` step of paper_v1:
graf reproduce configs/recipes/paper_v1.yaml --only compare

# Direct invocation of the harness for ablation variants:
python scripts/compare_baselines.py ... --models rf                  # all 42 features
python scripts/compare_baselines.py ... --models single_feature \
    --single-feature-name edge_attr_nonzero_frac                      # cue alone
python scripts/compare_baselines.py ... --models rf \
    --exclude-features edge_attr_nonzero_frac                         # RF minus cue
```

## Features

- **Detection & Tracking** – interfaces for YOLOv8, RT‑DETR, ByteTrack, BotSORT
- **Homography Calibration** – image→world coordinate transforms and ROI handling
- **Graph Construction** – spatial interaction graphs with class‑specific radii and kinematic edge features
- **Spatio‑Temporal Graphs** – rolling window graphs that connect actors across frames
- **Surrogate Safety Measures** – TTC, PET, DRAC, and event mining
- **Graph Models** – GCN, ST‑GCN, graph transformers, and classic baselines
- **Evaluation** – binary classification metrics, calibration, robustness analysis
- **Reproducible Pipeline** – configuration files, logging, experiment tracking, and CI

## CLI

The `graf` command is the entry point for the pipeline.

| Command | What it does |
|---|---|
| `graf doctor` | Check environment, data directories, detector weights, and recipes |
| `graf reproduce <recipe.yaml>` | Run a recipe file end-to-end |
| `graf detect-video --frames-dir X --model Y --output-dir Z` | Run a detector over a directory of frames |
| `graf fetch-detector --model YOLOv11-S` | Download UVH-26 detector weights into `data/models/` |
| `graf status` | Print pipeline status tree |
| `graf demo-graphs --outdir outputs/` | Write a toy PyG graph sample |
| `graf train-conflict-pairs --tracks ... --graphs_dir ... --homography_config ...` | Train a GCN on conflict-pair labels |

### Recipes

A recipe is a YAML file listing pipeline steps. Each step is a
subprocess command. Recipes are data-only; any non-trivial step lives
as a script under `scripts/` and is invoked from the recipe.

`configs/recipes/paper_v1.yaml` reproduces the paper's VNTraffic
primary result as five steps: fetch, prepare tracks, filter, build
graphs, compare.

```bash
graf reproduce configs/recipes/paper_v1.yaml              # run all steps
graf reproduce configs/recipes/paper_v1.yaml --dry-run    # preview
graf reproduce configs/recipes/paper_v1.yaml --only compare
graf reproduce configs/recipes/paper_v1.yaml --from-step build-graphs
```


### Test staging

Three tiers, by cost:

| Tier | When | What | Runtime |
|---|---|---|---|
| pre-commit | every `git commit` | hygiene, ruff, mypy, deptry | ~5-15 s |
| pre-push | every `git push` | + determinism, data-leakage, metamorphic, differential | +~10 s |
| CI | every push / PR | full suite (429 unit, 44 property) | ~2 min |

Skip the pre-push gate once with `git push --no-verify`.
### Local CI reproduction (tox)

Run the same checks CI runs, inside a fresh venv:

```bash
pip install tox
tox -e lint       # ruff check + ruff format --check + deptry  (~20 s)
tox -e unit       # pytest --cov, 308 tests                   (~3-5 min first run)
tox -e property   # Hypothesis under HYPOTHESIS_PROFILE=ci    (~3-5 min first run)
tox               # all three envs
```

Each env installs its own deps (`pip install -e .[dev]` plus the CPU torch
stack) so tox catches missing or misdeclared dependencies that a
long-lived dev environment has accreted. tox mirrors CI; CI does not
use tox — see `.github/workflows/ci.yml`.

## Installation

### CPU only

```bash
pip install -r requirements/base.txt -r requirements/dev.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric
