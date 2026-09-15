# MRC Intersection — smoke test

A stand-in clip used to verify the **detection and tracking** stages run
on real video. The target site for real validation is **NSC**.

**This is not a validation.** No homography, no ground truth, no
site-specific tuning, COCO-pretrained YOLOv8 rather than an Indian-traffic
model. See `results.md` for numbers and the honest list of limitations.

## Reproduce

```bash
# (Video is on Google Drive, not in the repo.)
ffmpeg -y -ss 40 -t 10 -i MRC_Intersection.mp4 -vf fps=5 -q:v 3 frames/f_%05d.jpg
python scripts/run_detection.py \
    --frames_dir frames --output_dir detections --stride 2
python scripts/run_tracking.py \
    --detections detections/detections.jsonl --output_dir tracks
```

Outputs live outside the repo; only this directory is committed.
