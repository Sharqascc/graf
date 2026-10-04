# Reproducibility

What can be reproduced from a fresh clone, and what cannot.

## TL;DR

| layer | reproducible from a fresh clone? | why |
|---|---|---|
| library code (`src/graf/`) | yes | committed |
| test suite (`tests/`) | yes | committed; no external data needed |
| end-to-end synthetic pipeline | yes | generated in-process |
| paper results (`docs/paper/*.md`) | **no** | input data not committed |

The code is reproducible. The results are not, because the inputs are
not. This is the honest state and it is stated as a limitation in the
paper.

## What a fresh clone can do

```bash
git clone https://github.com/Sharqascc/graf.git
cd graf
pip install -e '.[dev]'
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric

# Run the whole test suite
pytest -q tests/ --ignore=tests/test_property_based.py
pytest -q tests/test_property_based.py

# Run the end-to-end synthetic pipeline (no external data)
pytest tests/test_end_to_end_synthetic.py -v
```

## What a fresh clone cannot do

Re-run `scripts/compare_baselines.py` on VNTraffic and reproduce the
numbers in `docs/paper/`. The three inputs it needs are gitignored
and were lost with an ephemeral runtime:

- `data/raw/vntraffic_homography.yaml`
- `data/interim/vntraffic_tracks_all85_min20.jsonl`
- `data/processed/graphs/vntraffic_all85/`

See `docs/dataset_protocol.md` for the full directory contract and
the "Known issues and caveats" section of `README.md` for the
reader-facing version.

## Regenerating the inputs

Given the source video at `data/raw/videos/<video_id>.mp4` and a
hand-calibrated homography config at
`data/raw/<video_id>_homography.yaml`, the pipeline in
`docs/pipeline.md` regenerates everything under `interim/` and
`processed/`:

```bash
# 1. detection + tracking -> tracks JSONL
python scripts/run_detection.py --config configs/detection/<site>.yaml
python scripts/run_tracking.py  --config configs/tracking/<site>.yaml

# 2. trajectories + graphs
python scripts/build_trajectories.py \
    --tracks data/interim/tracks/<video_id>.jsonl \
    --homography_config data/raw/<video_id>_homography.yaml \
    --output data/interim/trajectories/<video_id>.json
python scripts/build_graphs.py \
    --tracks data/raw/<video_id>_tracks.jsonl \
    --homography_config data/raw/<video_id>_homography.yaml \
    --output_dir data/processed/graphs/<video_id>

# 3. the 10-fold purged rerun
#    (see docs/preregistration_sustained_r5.md for the decision rule;
#     full flag list in README and in any outputs/<run>/comparison.json)
python scripts/compare_baselines.py ...
```

## Data provenance risk

`data/raw/videos/*` is not committed. It is the top of the dependency
tree; if lost, the results cannot be regenerated from the repo alone.
This is why the repo does not claim reproducibility of the paper
results — only of the code and the tests.

Future work: an artifact registry (DVC, S3, or Zenodo) with checksums
so `data/raw/videos/` and the derived artifacts have a durable home
outside an ephemeral runtime.

---

## Branch protection and merge enforcement

GitHub enforces required status checks on `main` only when the
repository is public or on a paid plan. While `Sharquascc/graf` is
private on the free plan, branch protection is unavailable and a
red CI check does not block a merge.

The six CI jobs defined in `.github/workflows/ci.yml` still run on
every pull request, and this repository's merge procedure refuses
to merge a PR unless all six report `success` or `skipped`. That
refusal is client-side: it lives in the tooling that opens and
merges PRs, not in GitHub's policy layer.

**Practical rule while the repo is private:** merge only when every
job on the PR's checks page is green. Do not use the web UI merge
button without first opening the PR's *Checks* tab and confirming
all six jobs have passed.

**Enforcement returns automatically** when the repository is made
public at submission time. At that point, re-add the required
checks under Settings -> Branches -> Branch protection rules:

```
1 · Ruff
2 · Unit tests
3 · Property tests (Hypothesis)
4 · Invariants (pre-push suite)
5 · Manuscript build
6 · Dependency review
```

Six jobs, not three. The last three were added after the initial
branch protection rule and are not covered by the older three-check
configuration; a protection rule copied from before that change
would not block on them.

---

## Local pre-commit hooks in ephemeral environments

In a Colab runtime, `.git/hooks/` is wiped on every reset, so the
hooks declared in `.pre-commit-config.yaml` do nothing until they
are reinstalled. After each fresh clone, run:

```bash
bash scripts/setup_colab_hooks.sh
```

That installs both hook stages:

- `pre-commit` stage: ruff, ruff-format, mypy, import-linter,
  deptry, hygiene hooks (whitespace, EOF, YAML/TOML/AST, etc.).
- `pre-push` stage: the full unit-test suite except the Hypothesis
  property file (~78 s, 583 tests).

Local hooks are an early-warning layer, not a security boundary.
`git push --no-verify` skips every hook. The authoritative gate is
GitHub CI, which runs all six jobs on every pull request and
cannot be bypassed. See the Branch protection section above for
the current enforcement caveat.
