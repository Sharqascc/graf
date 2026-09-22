# Annotation protocol

This repo does not perform human annotation. Window labels are derived
from trajectory data via the rules in `scripts/label_strategies.py`.
What follows is the contract for the trajectory inputs the pipeline
consumes and the label definitions it produces.

## Trajectory input format

Tracked actors enter the pipeline as JSONL, one record per track:

    {"track_id": int, "video_id": str, "frames": [int, ...],
     "xyxy": [[x1, y1, x2, y2], ...], "class": str}

Required fields:

- `track_id` — unique within a video
- `video_id` — matches the homography config stem
- `frames` — ascending frame indices, one per detection
- `xyxy` — pixel-space bounding boxes aligned with `frames`
- `class` — one of `car`, `motorbike`, `bus`, `truck`, `person`

This matches the MOT-format conversion in `scripts/prepare_vntraffic.py`.

## Label definitions

Window labels are **derived**, not annotated. Strategies live in
`scripts/label_strategies.py`:

| strategy | rule |
|---|---|
| `any` | window positive if any frame contains a critical pair |
| `center` | window positive iff the middle frame is critical |
| `majority` | window positive iff >50% of frames are critical |
| `sustained` | window positive iff `run_length` consecutive frames are critical |
| `pair_fraction` | window positive iff >= `fraction_threshold` of critical pairs persist |

"Critical" is defined per SSM:

- **TTC** — `0 < ttc <= ttc_threshold_seconds` within `ttc_distance_threshold` metres
- **PET** — overlapping zone occupancy within `pet_threshold_seconds`
- **DRAC** — required deceleration >= `drac_threshold_mps2`

The paper's declared setup uses `sustained` with `run_length=5`; see
`docs/preregistration_sustained_r5.md`.

## What is not annotated

- Actor class is taken from the detector's COCO class, not corrected by hand.
- Occlusion, truncation, and detection confidence are not ground-truthed.
- There is no per-frame "criticality" label — the label is window-level
  and derived.

## Known limitation

Finding #9 of `docs/external_review_2026_09.md` notes that `sustained`
requires only that *some* critical pair exists in each of `run_length`
consecutive frames — not that the *same* pair persists. The label is
more accurately "a run of frames where some conflict exists" than "one
persisting conflict." A `label_persistent_pair` strategy is not
implemented; the caveat is documented, not fixed.

## Cross-reference

- `scripts/label_strategies.py` — strategy implementations
- `scripts/evaluate_vntraffic.py` — `build_labels`, TTC path
- `docs/external_review_2026_09.md` #9 — pair-identity caveat
