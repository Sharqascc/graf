"""Evaluate YOLO detection against hand-annotated ground truth.

Runs a YOLO model on the frames of a VNTraffic-style clip, matches
predictions to the MOT-format ground-truth boxes by IoU, and reports
COCO-style average precision at IoU thresholds 0.5 and 0.5:0.95.

Class-agnostic matching by default, because the VNTraffic ground truth
carries no class labels. Optionally restrict YOLO predictions to a set
of vehicle classes so that pedestrians (which the ground truth does not
annotate) are not counted as false positives.

Usage:
    python scripts/evaluate_detection.py \
        --frames_dir data/interim/frames/vntraffic/VNTraffic_Original-video \
        --gt_txt data/external/vntraffic/VNTraffic/VNTraffic_GroundTruth.txt \
        --model yolov8n.pt \
        --output_dir outputs/detection_eval \
        --report docs/paper/detection_metrics.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# COCO class ids that correspond to vehicles. Ultralytics YOLO uses COCO.
VEHICLE_CLASSES = {2, 3, 5, 7}  # car, motorcycle, bus, truck


def parse_mot_gt(path: Path) -> dict[int, list[tuple[float, float, float, float]]]:
    """Parse MOT ground truth into {frame_idx: [xyxy, ...]}.

    MOT columns: frame, id, x, y, w, h, conf, ...
    The x, y are top-left; w, h are width, height.
    """
    gt: dict[int, list[tuple[float, float, float, float]]] = defaultdict(list)
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            frame = int(parts[0])
            x, y, w, h = (
                float(parts[2]),
                float(parts[3]),
                float(parts[4]),
                float(parts[5]),
            )
            gt[frame].append((x, y, x + w, y + h))
    return dict(gt)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every row of a (M, 4) and every row of b (N, 4)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)), dtype=float)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def match_frame(preds: np.ndarray, gt: np.ndarray, iou_thr: float) -> np.ndarray:
    """Return a boolean array: which predictions are TPs.

    Greedy matching by descending IoU, one-to-one between preds and GT.
    """
    if preds.size == 0 or gt.size == 0:
        return np.zeros(len(preds), dtype=bool)
    ious = iou_matrix(preds, gt)
    matched_gt = np.zeros(len(gt), dtype=bool)
    is_tp = np.zeros(len(preds), dtype=bool)
    # iterate over preds in order of their best unmatched IoU
    order = np.argsort(-ious.max(axis=1))
    for pi in order:
        candidates = ious[pi]
        # consider only unmatched GT
        candidates = np.where(matched_gt, -1.0, candidates)
        if candidates.size == 0:
            continue
        gi = int(np.argmax(candidates))
        if candidates[gi] >= iou_thr:
            matched_gt[gi] = True
            is_tp[pi] = True
    return is_tp


def compute_ap(
    preds_by_frame: dict[int, np.ndarray],
    gt_by_frame: dict[int, np.ndarray],
    iou_thr: float,
) -> dict:
    """COCO-style average precision at a single IoU threshold.

    Returns dict with AP, precision, recall, F1 at the operating point
    that maximises F1.
    """
    # collect (conf, is_tp) over all frames
    all_conf: list[float] = []
    all_tp: list[int] = []
    n_gt = sum(len(v) for v in gt_by_frame.values())

    for frame in sorted(gt_by_frame.keys()):
        gt = gt_by_frame[frame]
        preds_conf = preds_by_frame.get(frame, np.zeros((0, 5), dtype=float))
        if preds_conf.shape[0] == 0:
            all_conf.extend([])
            all_tp.extend([])
            continue
        # sort preds by confidence descending; matching uses this order
        order = np.argsort(-preds_conf[:, 4])
        preds_conf = preds_conf[order]
        xyxy = preds_conf[:, :4]
        is_tp = match_frame(
            xyxy, np.asarray(gt) if len(gt) else np.zeros((0, 4)), iou_thr
        )
        all_conf.extend(preds_conf[:, 4].tolist())
        all_tp.extend(is_tp.astype(int).tolist())

    if not all_conf:
        return {"AP": 0.0, "precision": 0.0, "recall": 0.0, "F1": 0.0, "n_gt": n_gt}

    confs = np.asarray(all_conf)
    tps = np.asarray(all_tp, dtype=int)
    order = np.argsort(-confs)
    tps = tps[order]
    cum_tp = np.cumsum(tps)
    cum_fp = np.cumsum(1 - tps)
    precision = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)
    recall = cum_tp / max(n_gt, 1)

    # 101-point interpolated AP (COCO style)
    recall_thresholds = np.linspace(0, 1, 101)
    prec_interp = []
    for r in recall_thresholds:
        mask = recall >= r
        prec_interp.append(float(precision[mask].max()) if mask.any() else 0.0)
    AP = float(np.mean(prec_interp))

    # F1-optimal operating point
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    best = int(np.argmax(f1))
    return {
        "AP": AP,
        "precision": float(precision[best]),
        "recall": float(recall[best]),
        "F1": float(f1[best]),
        "n_gt": int(n_gt),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--gt_txt", required=True)
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Low threshold so the PR curve is fully sampled",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--output_dir", default="outputs/detection_eval")
    ap.add_argument(
        "--report", default=None, help="If set, write a markdown report here"
    )
    ap.add_argument(
        "--vehicle_only",
        action="store_true",
        help="Restrict predictions to COCO vehicle classes",
    )
    args = ap.parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print(
            "ultralytics not installed. Run: pip install ultralytics", file=sys.stderr
        )
        return 2

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"frames dir not found: {frames_dir}", file=sys.stderr)
        return 2

    gt = parse_mot_gt(Path(args.gt_txt))
    print(f"ground truth: {len(gt)} frames, {sum(len(v) for v in gt.values())} boxes")

    model = YOLO(args.model)
    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    print(f"frames: {len(frame_paths)}")

    preds_by_frame: dict[int, np.ndarray] = {}
    gt_by_frame: dict[int, np.ndarray] = {}
    skipped_no_gt = 0

    for fp in frame_paths:
        try:
            frame_idx = int(fp.stem)
        except ValueError:
            continue
        if frame_idx not in gt:
            skipped_no_gt += 1
            continue
        results = model.predict(
            source=str(fp),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device="cpu",
        )
        r = results[0]
        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            preds_by_frame[frame_idx] = np.zeros((0, 5), dtype=float)
        else:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
            cls = boxes.cls.cpu().numpy().astype(int)
            if args.vehicle_only:
                keep = np.isin(cls, list(VEHICLE_CLASSES))
                xyxy = xyxy[keep]
                conf = conf[keep]
            preds_by_frame[frame_idx] = np.concatenate([xyxy, conf], axis=1)
        gt_by_frame[frame_idx] = np.asarray(gt[frame_idx], dtype=float)

    print(f"processed {len(preds_by_frame)} frames ({skipped_no_gt} skipped for no GT)")

    # AP at each IoU threshold
    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    per_iou = {}
    for t in iou_thresholds:
        per_iou[float(t)] = compute_ap(preds_by_frame, gt_by_frame, float(t))

    ap50 = per_iou[0.5]["AP"]
    ap5095 = float(np.mean([per_iou[float(t)]["AP"] for t in iou_thresholds]))

    payload = {
        "model": args.model,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "vehicle_only": args.vehicle_only,
        "num_frames_evaluated": len(preds_by_frame),
        "num_gt_boxes": int(sum(len(v) for v in gt_by_frame.values())),
        "AP@0.5": ap50,
        "AP@0.5:0.95": ap5095,
        "per_iou": {f"{t:.2f}": v for t, v in per_iou.items()},
        "operating_point_at_AP50": {
            "precision": per_iou[0.5]["precision"],
            "recall": per_iou[0.5]["recall"],
            "F1": per_iou[0.5]["F1"],
        },
    }

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "detection_metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {outdir / 'detection_metrics.json'}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Detection metrics — YOLO on VNTraffic",
            "",
            f"**Model:** {args.model}  ",
            f"**Image size:** {args.imgsz}  ",
            f"**Confidence threshold:** {args.conf}  ",
            f"**Vehicle classes only:** {args.vehicle_only}  ",
            f"**Frames evaluated:** {len(preds_by_frame)}  ",
            f"**Ground-truth boxes:** {int(sum(len(v) for v in gt_by_frame.values()))}  ",
            "",
            "## Summary",
            "",
            f"- **AP@0.5:** {ap50:.4f}",
            f"- **AP@0.5:0.95:** {ap5095:.4f}",
            f"- Precision at F1-optimal point (IoU=0.5): "
            f"{per_iou[0.5]['precision']:.4f}",
            f"- Recall at F1-optimal point (IoU=0.5): {per_iou[0.5]['recall']:.4f}",
            f"- F1 at F1-optimal point (IoU=0.5): {per_iou[0.5]['F1']:.4f}",
            "",
            "## Per-IoU-threshold AP",
            "",
            "| IoU | AP | precision | recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
        for t in iou_thresholds:
            r = per_iou[float(t)]
            lines.append(
                f"| {t:.2f} | {r['AP']:.4f} | {r['precision']:.4f} | "
                f"{r['recall']:.4f} | {r['F1']:.4f} |"
            )
        lines += [
            "",
            "## Method",
            "",
            "Class-agnostic matching by IoU, greedy one-to-one per frame. "
            "AP computed COCO-style (101-point interpolated). Predictions "
            "run at a low confidence threshold (0.001) so the full "
            "precision-recall curve is sampled. The reported precision, "
            "recall, and F1 are taken at the operating point that "
            "maximizes F1 on the curve.",
            "",
            "The ground-truth VNTraffic MOT file does not carry class "
            "labels; every annotated actor is a vehicle. If "
            "`--vehicle_only` is set, YOLO predictions in pedestrian "
            "class are dropped before matching, so pedestrians do not "
            "count as false positives.",
        ]
        report_path.write_text("\n".join(lines) + "\n")
        print(f"wrote {report_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
