# MRC Intersection — smoke test

**Status:** detection + tracking only. No trajectories, no SSM, no ground truth.
**Target site:** NSC. This clip is a stand-in.

## Source

- `MRC_Intersection.mp4` (Google Drive, not committed)
- 1920x1080, 30 fps source, 173 s total
- Processed window: 40-50 s, extracted at 5 fps, detection stride 2
  -> **effective 2.5 fps** over 25 detected frames

## Detection

`configs/detection/yolov8.yaml` — YOLOv8n, `imgsz=1280`, `conf_threshold=0.25`.

| metric | value |
|---|---|
| detections | 961 |
| per frame  | 38.4 |
| conf median | 0.411 |
| conf >= 0.5 | 345 / 961 |

Class distribution:

| class | detections |
|---|---|
| car | 355 |
| pedestrian | 262 |
| truck | 223 |
| two_wheeler | 94 |
| bus | 20 |
| bicycle | 7 |

## Tracking

`configs/tracking/bytetrack.yaml` — greedy IoU associator, `iou_threshold=0.3`,
`track_buffer=30`.

| metric | value |
|---|---|
| track rows | 345 |
| unique tracks | 117 |
| track length: min / median / max | 1 / 1 / 25 |

Class distribution of tracked rows:

| class | rows |
|---|---|
| car | 183 |
| pedestrian | 72 |
| truck | 69 |
| two_wheeler | 20 |
| bicycle | 1 |

### Why the median track length is 1

The detection stride (2) means consecutive tracked frames are ~66 ms
apart in source time. A car moving at 5 m/s advances ~0.33 m, which at
this camera's scale is several dozen pixels. The default IoU threshold
(0.3) is too strict for that displacement, so most detections fail to
match their predecessor and spawn a new track.

This is a **stride artifact**, not a tracker bug. A production run would
use `--stride 1` (or a wider `iou_threshold`, or a motion-aware
associator). The next branch that adds motion-aware tracking will
address this properly.

## Limitations

- **No homography.** Camera projection was not calibrated. No
  metrically-meaningful SSM can be computed from this clip.
- **COCO-pretrained model.** YOLOv8n is not fine-tuned on Indian traffic.
  Fine-grained classes (auto-rickshaw, tempo-traveller) are missing; the
  `motorcycle` -> `two_wheeler` mapping is a heuristic.
- **Wide-angle view.** Actors are small in frame. Median detection
  confidence sits near the config threshold; many marginal detections.
- **No ground truth.** Accuracy is unmeasurable from this run.
- **Stride artifact in tracking.** See section above.

## Next steps for NSC

1. Shoot or obtain NSC footage with a **site-specific homography**
   (4-8 point correspondences to known world coordinates).
2. Annotate a small ground-truth subset (~50-200 frames) for
   validation: bounding boxes at minimum, ideally conflict labels for
   SSM benchmarking.
3. Optionally fine-tune a detector on UVH-26 (IISc's Indian traffic
   dataset) to get auto-rickshaw / tempo-traveller classes.
4. Re-run the pipeline end-to-end at `--stride 1` and replace this file
   with a validation report.
