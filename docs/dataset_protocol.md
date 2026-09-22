# Dataset protocol

## Directory structure

    data/
      raw/          # immutable inputs; some gitignored
        videos/     # source video files                     (gitignored)
        metadata/   # site notes, camera notes               (committed)
        site_notes/ # per-site calibration notes             (committed)
        *.yaml      # homography configs                     (gitignored)
        *.jsonl     # prepared tracks                        (gitignored)
      interim/      # derived, regenerable                   (all gitignored)
        detections/ # per-frame detection output
        tracks/     # per-track trajectories
        trajectories/ # world-space trajectories
        homography/ # computed homographies
        frames/     # decoded frames
      processed/    # model-ready, regenerable               (all gitignored)
        graphs/     # per-frame PyG graph objects
        windows/    # temporal window datasets
        ssm_events/ # mined conflict events
        labels/     # derived window labels
      external/     # third-party datasets                   (all gitignored)

## Committed vs generated

| path | committed | regenerable from |
|---|---|---|
| `data/raw/videos/*` | no | source (unrecoverable if lost) |
| `data/raw/*.yaml` | no | hand calibration + site notes |
| `data/raw/*.jsonl` | no | `data/raw/videos/` + detection + tracking |
| `data/interim/*` | no | `data/raw/` + pipeline scripts |
| `data/processed/*` | no | `data/interim/` + pipeline scripts |
| `data/raw/metadata/*` | yes | — |
| `data/raw/site_notes/*` | yes | — |

The pipeline is designed so that everything under `interim/` and
`processed/` is regenerable from `raw/`. Anything under `raw/` that is
not committed is a hard dependency of the paper's numbers — see the
reproducibility section of `README.md` and `docs/reproducibility.md`.

## Naming

- `video_id` is the stem of the source video file. It keys configs,
  tracks, graphs, and metrics across the pipeline.
- Homography configs live at `data/raw/<video_id>_homography.yaml`
  unless a different path is passed with `--homography_config`.
- Per-video outputs live at `data/processed/<stage>/<video_id>/`.

## Cleaning

- `make clean-interim` removes `interim/` and `processed/` without
  touching `raw/`.
- `make clean-all` also removes downloaded `external/` data.
- Neither target removes anything committed to git.

## Known limitation

As of 2026-09 the `vntraffic` raw inputs
(`data/raw/vntraffic_homography.yaml`,
`data/interim/vntraffic_tracks_all85_min20.jsonl`,
`data/processed/graphs/vntraffic_all85/`) were lost with an ephemeral
runtime. `README.md` records this. Regeneration requires the source
video, which was not itself committed.

## Cross-reference

- `docs/pipeline.md` — the stages that populate each directory
- `docs/reproducibility.md` — the regenerate-from-raw walkthrough
- `.gitignore` — the specific paths excluded from version control
