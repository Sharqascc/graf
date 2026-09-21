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
# RF on all features
python scripts/compare_baselines.py ... --models rf

# The trivial cue alone
python scripts/compare_baselines.py ... --models single_feature \
    --single-feature-name edge_attr_nonzero_frac

# RF without the trivial cue
python scripts/compare_baselines.py ... --models rf \
    --exclude-features edge_attr_nonzero_frac
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

```bash
graf status                              # print pipeline status tree
graf demo-graphs --outdir outputs/       # write a toy PyG graph sample
graf train-conflict-pairs --tracks <tracks.jsonl> --graphs_dir <graphs> \
    --homography_config <h.yaml> --output_dir outputs/models_conflict_pairs \
    --epochs 50 --num_folds 5 --seed 42
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
