# Detection metrics — YOLO on VNTraffic

- **Model:** data/models/UVH-26-MV-YOLOv11-S.pt
- **Image size:** 640
- **Confidence threshold:** 0.001
- **Vehicle classes only:** False
- **Frames evaluated:** 501
- **Ground-truth boxes:** 16036

## Summary

- **AP@0.5:** 0.6231
- **AP@0.5:0.95:** 0.2412
- Precision at F1-optimal point (IoU=0.5): 0.5927
- Recall at F1-optimal point (IoU=0.5): 0.6785
- F1 at F1-optimal point (IoU=0.5): 0.6327

## Per-IoU-threshold AP

| IoU | AP | precision | recall | F1 |
|---|---:|---:|---:|---:|
| 0.50 | 0.6231 | 0.5927 | 0.6785 | 0.6327 |
| 0.55 | 0.5109 | 0.5238 | 0.6239 | 0.5695 |
| 0.60 | 0.3830 | 0.4531 | 0.5493 | 0.4966 |
| 0.65 | 0.2573 | 0.3867 | 0.4688 | 0.4238 |
| 0.70 | 0.1805 | 0.3207 | 0.4309 | 0.3677 |
| 0.75 | 0.1506 | 0.2960 | 0.3973 | 0.3392 |
| 0.80 | 0.1283 | 0.2783 | 0.3609 | 0.3143 |
| 0.85 | 0.0977 | 0.2500 | 0.3028 | 0.2738 |
| 0.90 | 0.0632 | 0.1975 | 0.2369 | 0.2154 |
| 0.95 | 0.0180 | 0.0779 | 0.0920 | 0.0844 |

## Method

Class-agnostic matching by IoU, greedy one-to-one per frame. AP computed COCO-style (101-point interpolated). Predictions run at a low confidence threshold (0.001) so the full precision-recall curve is sampled. The reported precision, recall, and F1 are taken at the operating point that maximizes F1 on the curve.

The ground-truth VNTraffic MOT file does not carry class labels; every annotated actor is a vehicle. If `--vehicle_only` is set, YOLO predictions in pedestrian class are dropped before matching, so pedestrians do not count as false positives.
