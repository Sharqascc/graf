# Detection on MRC Intersection (UVH-26)

**Detector:** UVH-26-MV-YOLOv11-S (IISc Bengaluru, arXiv:2511.02563)  
**Clip:** MRC Intersection (Vadodara), window 40-70s at 5 fps  
**Input:** imgsz=1280, confidence threshold 0.50  
**Frames evaluated:** 150

## Summary

| metric | value |
|---|---:|
| mean detections / frame | 48.36 +/- 3.96 |
| min / max detections | 37 / 57 |
| mean confidence | 0.781 |

## Class distribution

| class | detections |
|---|---:|
| Two-wheeler | 3,163 |
| Three-wheeler | 2,909 |
| Hatchback | 449 |
| Sedan | 302 |
| Truck | 150 |
| LCV | 139 |
| Van | 92 |
| bicycle | 25 |
| MUV | 23 |
| SUV | 1 |
| Bus | 1 |

84% of detections are Two-wheeler or Three-wheeler, consistent with
the observed traffic mix at the site. All detections fall into Indian
vehicle classes; UVH-26's 14-class taxonomy covers the scene without
recourse to an Others fallback.

## Why UVH-26

1. **Taxonomy.** COCO has no class for auto-rickshaw, tempo-traveller,
   LCV, or mini-bus. Assigning those to car or truck loses the
   distinction that matters for conflict analysis.
2. **Domain match.** UVH-26 was trained on 26,646 frames of Indian
   CCTV traffic. All clips in this repository are Indian.
3. **Performance.** At 1080p input and conf >= 0.50, UVH-26 produces
   stable per-frame counts (std 3.96 on a mean of 48.36) and high
   mean confidence (0.781).

## Caveats

- **No ground truth on MRC.** Detection quality at the reporting
  threshold is verified by visual inspection of annotated samples
  (figures/mrc_uvh26_sample_*.png), not by a measured metric.
- **False positives on static infrastructure.** Visual inspection
  at lower thresholds (0.25-0.40) revealed occasional detections on
  concrete pillars and road surface. These do not survive the 0.50
  threshold in the sampled frames, but the exact false-positive rate
  is not measured.
- **Single clip.** One 30-second window at one intersection.

## Reproduce

```bash
python scripts/fetch_detector.py --model YOLOv11-S
ffmpeg -y -ss 40 -t 30 -i data/raw/videos/mrc_intersection.mp4 \
    -vf fps=5 -q:v 3 data/interim/frames/mrc_40_70s/f_%05d.jpg
python scripts/detect_video.py \
    --frames_dir data/interim/frames/mrc_40_70s \
    --model data/models/UVH-26-MV-YOLOv11-S.pt \
    --output_dir outputs/mrc_uvh26_1280_conf50 \
    --samples 6 --conf 0.50 --imgsz 1280
```
