# Quality metrics

Regenerated after deleting the 33 empty scaffolding files and the
two dead packages (`detection/`, `visualization/`).

## Summary

| metric | value |
|---|---:|
| Unit tests | 583 passed, 3 xfailed |
| Property tests (Hypothesis) | 48 passed |
| Coverage (src + scripts) | 76% |
| ruff check | PASS |
| ruff format | PASS |

## Continuous integration gates

Every PR must pass all six:

| job | what it runs |
|---|---|
| `1 · Ruff` | ruff, ruff-format, mypy, import-linter, deptry, hygiene hooks |
| `2 · Unit tests` | 583 tests + coverage floor 65% |
| `3 · Property tests` | 48 Hypothesis tests (ci profile) |
| `4 · Invariants` | determinism, data-leakage, input-boundaries, metamorphic, differential |
| `5 · Manuscript build` | `docs/paper/manuscript.md` in sync with section sources |
| `6 · Dependency review` | advisory CVE scan on PRs |

Local pre-push hook matches jobs 2-3 minus property tests: runs the
same unit suite plus the coverage floor (~105 s).

Nightly: deep Hypothesis profile (1000 examples), mutation diagnostic on `ssm/ttc.py`.
Weekly: Dependabot opens pip and GitHub Actions update PRs.

## Repository structure

```
src/graf/
  calibration/    homography, world coordinates
  data/           schema, dataset loaders, graph windows
  evaluation/     binary metrics
  graph/          edge/node features, builders, temporal window
  models/         baseline classifiers and GCN
  ssm/            TTC, DRAC, PET, event mining
  tracking/       ByteTrack / BoT-SORT wrappers
  training/       conflict-pair training loop
  trajectories/   kinematics, conflict pairs
  utils/          seeds, io, logging, pipeline status
scripts/          CLI entry points for each pipeline stage
tests/            unit + property tests
docs/paper/       manuscript sections, frozen results, references
```

## CI job list to add to branch protection when the repo becomes public

```
1 · Ruff
2 · Unit tests
3 · Property tests (Hypothesis)
4 · Invariants (pre-push suite)
5 · Manuscript build
6 · Dependency review
```

## Honest limitations

- Branch protection is not available on a private repo on the free plan.
  See `docs/reproducibility.md` for the caveat and the mitigation.
- `--no-verify` bypasses local hooks; only GitHub CI cannot be bypassed.
- Label TTC differs from `ssm/ttc.py` by design; documented and tested.
- Two citations carry known errors, marked `[CHECK]` in `references.md`.
