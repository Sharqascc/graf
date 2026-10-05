# Detection metrics — YOLO on VNTraffic

- **Model:** yolov8n.pt
- **Image size:** 640
- **Confidence threshold:** 0.001
- **Vehicle classes only:** True
- **Frames evaluated:** 501
- **Ground-truth boxes:** 16036

## Summary

- **AP@0.5:** 0.6819
- **AP@0.5:0.95:** 0.4749
- Precision at F1-optimal point (IoU=0.5): 0.6577
- Recall at F1-optimal point (IoU=0.5): 0.6036
- F1 at F1-optimal point (IoU=0.5): 0.6295

## Per-IoU-threshold AP

| IoU | AP | precision | recall | F1 |
|---|---:|---:|---:|---:|
| 0.50 | 0.6819 | 0.6577 | 0.6036 | 0.6295 |
| 0.55 | 0.6714 | 0.6525 | 0.5988 | 0.6245 |
| 0.60 | 0.6548 | 0.6415 | 0.5887 | 0.6140 |
| 0.65 | 0.6232 | 0.6772 | 0.5307 | 0.5951 |
| 0.70 | 0.5762 | 0.6720 | 0.4963 | 0.5709 |
| 0.75 | 0.5061 | 0.6790 | 0.4377 | 0.5323 |
| 0.80 | 0.4286 | 0.6554 | 0.3901 | 0.4891 |
| 0.85 | 0.3367 | 0.6463 | 0.3192 | 0.4274 |
| 0.90 | 0.2143 | 0.5149 | 0.2327 | 0.3206 |
| 0.95 | 0.0561 | 0.2532 | 0.0877 | 0.1302 |

## Method

Class-agnostic matching by IoU, greedy one-to-one per frame. AP computed COCO-style (101-point interpolated). Predictions run at a low confidence threshold (0.001) so the full precision-recall curve is sampled. The reported precision, recall, and F1 are taken at the operating point that maximizes F1 on the curve.

The ground-truth VNTraffic MOT file does not carry class labels; every annotated actor is a vehicle. If `--vehicle_only` is set, YOLO predictions in pedestrian class are dropped before matching, so pedestrians do not count as false positives.
