# Quality metrics

**Generated from a live measurement of the repository at each commit.**
Regenerate with the cell documented in `docs/reproducibility.md`.

## Summary

| metric | value |
|---|---:|
| Unit tests | 583 passed, 3 xfailed |
| Property tests (Hypothesis) | 48 passed |
| Test files | 59 |
| Source lines | 5,324 |
| Test lines | 9,813 |
| Coverage (src + scripts) | 66% |
| ruff check | PASS |
| ruff format | PASS |
| TODO / FIXME markers | 0 |

## Continuous integration gates

Every PR must pass all six:

| job | what it runs |
|---|---|
| `1 · Ruff` | ruff, ruff-format, mypy, import-linter, deptry, hygiene hooks (13 total) |
| `2 · Unit tests` | 583 tests + coverage floor 65% |
| `3 · Property tests` | 48 Hypothesis tests, `ci` profile |
| `4 · Invariants` | determinism, data-leakage, input-boundaries, metamorphic, differential |
| `5 · Manuscript build` | `docs/paper/manuscript.md` in sync with section sources |
| `6 · Dependency review` | advisory CVE scan on PRs (GitHub dependency graph) |

Nightly: deep Hypothesis profile (1000 examples), mutation diagnostic on `ssm/ttc.py`.
Weekly: Dependabot opens pip and GitHub Actions update PRs.

## Per-module coverage

Lowest first. Modules at 0% are either placeholder stubs or
build-time tools excluded from the coverage floor.

| module | stmts | miss | cover |
|---|---:|---:|---:|
| `scripts/bootstrap_repo.py` | 6 | 6 | 0% |
| `scripts/build_graph_dataset.py` | 37 | 37 | 0% |
| `scripts/build_graphs.py` | 63 | 63 | 0% |
| `scripts/estimate_homography.py` | 56 | 56 | 0% |
| `scripts/evaluate_gcn_risk.py` | 54 | 54 | 0% |
| `scripts/evaluate_model.py` | 23 | 23 | 0% |
| `scripts/export_graph_samples.py` | 16 | 16 | 0% |
| `scripts/export_paper_results.py` | 76 | 76 | 0% |
| `scripts/extract_frames.py` | 32 | 32 | 0% |
| `scripts/fetch_vntraffic.py` | 41 | 41 | 0% |
| `scripts/freeze_lock.py` | 64 | 64 | 0% |
| `scripts/label_sensitivity.py` | 202 | 202 | 0% |
| `scripts/make_windows.py` | 30 | 30 | 0% |
| `scripts/mutate_ttc.py` | 66 | 66 | 0% |
| `scripts/pipeline_status.py` | 90 | 90 | 0% |
| `scripts/prepare_vntraffic.py` | 82 | 82 | 0% |
| `scripts/train_gcn_risk.py` | 175 | 175 | 0% |
| `scripts/train_model.py` | 18 | 18 | 0% |
| `scripts/run_detection.py` | 72 | 45 | 38% |
| `src/graf/training/conflict_pairs.py` | 99 | 55 | 44% |
| `scripts/evaluate_vntraffic.py` | 287 | 121 | 58% |
| `scripts/run_tracking.py` | 176 | 44 | 75% |
| `scripts/compare_baselines.py` | 309 | 71 | 77% |
| `src/graf/models/gcn_risk.py` | 69 | 14 | 80% |
| `scripts/train_conflict_pairs.py` | 12 | 2 | 83% |
| `src/graf/data/dataset.py` | 35 | 5 | 86% |
| `scripts/compute_ssm.py` | 130 | 16 | 88% |
| `scripts/make_synthetic_dataset.py` | 89 | 10 | 89% |
| `src/graf/graph/features.py` | 197 | 22 | 89% |
| `src/graf/graph/nodes.py` | 126 | 14 | 89% |
| `src/graf/utils/seeds.py` | 19 | 2 | 89% |
| `src/graf/graph/builders.py` | 276 | 25 | 91% |
| `src/graf/graph/edges.py` | 82 | 7 | 91% |
| `src/graf/graph/pyg_export.py` | 101 | 9 | 91% |
| `src/graf/graph/temporal.py` | 114 | 10 | 91% |
| `scripts/mine_ssm_events.py` | 60 | 5 | 92% |
| `src/graf/utils/io.py` | 52 | 4 | 92% |
| `src/graf/evaluation/binary_metrics.py` | 29 | 2 | 93% |
| `src/graf/utils/pipeline_status.py` | 83 | 6 | 93% |
| `src/graf/models/baselines.py` | 286 | 17 | 94% |
| ... and 59 more | | | |

### Untested modules (0%)

| module | stmts | note |
|---|---:|---|
| `scripts/bootstrap_repo.py` | 6 | candidate for coverage or deletion |
| `scripts/build_graph_dataset.py` | 37 | candidate for coverage or deletion |
| `scripts/build_graphs.py` | 63 | candidate for coverage or deletion |
| `scripts/estimate_homography.py` | 56 | candidate for coverage or deletion |
| `scripts/evaluate_gcn_risk.py` | 54 | candidate for coverage or deletion |
| `scripts/evaluate_model.py` | 23 | candidate for coverage or deletion |
| `scripts/export_graph_samples.py` | 16 | candidate for coverage or deletion |
| `scripts/export_paper_results.py` | 76 | candidate for coverage or deletion |
| `scripts/extract_frames.py` | 32 | candidate for coverage or deletion |
| `scripts/fetch_vntraffic.py` | 41 | candidate for coverage or deletion |
| `scripts/freeze_lock.py` | 64 | candidate for coverage or deletion |
| `scripts/label_sensitivity.py` | 202 | candidate for coverage or deletion |
| `scripts/make_windows.py` | 30 | candidate for coverage or deletion |
| `scripts/mutate_ttc.py` | 66 | candidate for coverage or deletion |
| `scripts/pipeline_status.py` | 90 | candidate for coverage or deletion |
| `scripts/prepare_vntraffic.py` | 82 | candidate for coverage or deletion |
| `scripts/train_gcn_risk.py` | 175 | candidate for coverage or deletion |
| `scripts/train_model.py` | 18 | candidate for coverage or deletion |

## Test inventory

| file | purpose |
|---|---|
| `test_compare_baselines.py` | CV harness, model comparison, calibration |
| `test_fixture_pipeline.py` | end-to-end smoke on the synthetic fixture |
| `test_pipeline_invariants.py` | leakage, purge, threshold objective invariants |
| `test_label_ttc_parity.py` | label TTC vs `ssm/ttc.py` — agreement and divergence |
| `test_label_persistent_pair.py` | same-pair label strategy |
| `test_rel_heading_diagnostic.py` | documents the constant-zero heading column |
| `test_determinism.py` | bitwise identical output under repeated calls |
| `test_data_leakage.py` | cross-fold frame overlap bounds |
| `test_input_boundaries.py` | malformed input degrades, does not crash |
| `test_metamorphic.py` | translation, rotation invariance of SSM |
| `test_differential.py` | closed-form vs naive numeric agreement |
| `test_property_based.py` | Hypothesis invariants across SSM and graph builders |
| `test_numerical_stability.py` | physical bounds and edge cases |
| `test_ssm*.py` | per-indicator unit tests |
| `test_golden_master.py` | frozen reference values |

## Known gaps

- **Coverage floor is 65%.** Thin margin; adding untested files will drop it.
  Three build tools are explicitly omitted (`make_fixture.py`, `build_manuscript.py`,
  and the deleted `train_tabular_fold`).
- **Label TTC differs from `ssm/ttc.py`** by design. Documented and tested;
  a follow-up study could align them.
- **`rel_heading_sin` is dead on VNTraffic** (no heading input). Documented and tested.
- **AICC22 replication has 4 effective folds** (one fold has zero negatives).
- **Two citations carry known errors** (`McDermott et al.` year, `Zheng et al.` venue).
  Marked `[CHECK]` in `docs/paper/references.md`.

## Source code structure

```
src/graf/
  calibration/    homography, ROI, world coordinates
  data/           schema, dataset loaders, graph windows
  detection/      YOLO / RT-DETR wrappers (scaffold)
  evaluation/     binary metrics, calibration, error analysis
  graph/          edge/node features, builders, temporal window
  models/         baseline classifiers and GCN
  ssm/            TTC, DRAC, PET, event mining
  tracking/       ByteTrack / BoT-SORT wrappers
  training/       conflict-pair training loop
  trajectories/   kinematics, smoothing, interpolation
  utils/          seeds, io, logging, geometry
  visualization/  overlays, plots (not tested)
scripts/          CLI entry points for each pipeline stage
tests/            583 unit + 48 property tests
docs/paper/       manuscript sections, frozen results, references
```
