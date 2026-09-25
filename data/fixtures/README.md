# Synthetic fixture

A small, deterministic dataset that exercises the full `graf`
pipeline without the real VNTraffic data. Used by
`tests/test_fixture_pipeline.py` and available for anyone wanting
to run the harness end-to-end.

## Contents

    demo/
      tracks.jsonl       # 360 records: 60 frames x 6 actors
      homography.yaml    # identity (pixel coords == world coords)
      graphs/            # 60 per-frame PyG .pt files
      meta.json          # seed, fps, actor count

## Regenerating

    python scripts/make_fixture.py --out data/fixtures/demo

Deterministic: same seed, same inputs, same output. Byte-identical
on the same numpy version. The .pt files are committed despite the
blanket `*.pt` gitignore rule; `.gitignore` has an explicit
exception for `data/fixtures/**/*.pt`.

## What the fixture exercises

- JSONL tracks parsing + `filter_tracks` + `add_world_coords`
- Homography application (identity in this case)
- PyG graph construction via `build_pyg_graph_for_frame`
- Sustained-label computation (`_frame_ttc_stats` + `label_sustained`)
- `blocked_folds` + purge logic + `leakage_report`
- Nested-CV threshold calibration, including the single-class guard
  (fold 2 has zero negatives by construction, so the guard fires)
- `comparison.json` serialization

## What the fixture does NOT do

- Reproduce the paper's numbers. AUC is near 0.5 for all models on
  the fixture because the synthetic trajectories are trivial.
- Cover detection or tracking (the fixture starts from tracks, not
  raw video).
- Cover GPUs, real homographies, or more than one video.

## Design notes

- 60 frames at 30 fps gives 2 seconds; combined with window=5,
  stride=2 this produces 28 windows, 11 of which are positive under
  the sustained label at run_length=3.
- Two cars close head-on with 2 m lateral offset; their TTC drops
  below 1.5 s around frame 18 and stays there. The other four actors
  move away from each other and act as noise.
- num_folds=3 is the smallest sensible fold count for a smoke test;
  fold 2 gets zero validation negatives by design, which triggers
  the single-class guard in `evaluate_model`. If that guard
  regresses, the smoke test catches it.
