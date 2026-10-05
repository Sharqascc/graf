"""Run detection on a directory of frames, no ground truth needed.

Produces per-frame detections, aggregate class statistics, and annotated
sample frames for visual inspection. Useful when there is no hand
annotation to evaluate against, as with our own test footage.

Usage:
    python scripts/detect_video.py \
        --frames_dir data/interim/frames/sama/Sama_Test_Video \
        --model data/models/UVH-26-MV-YOLOv11-S.pt \
        --output_dir outputs/sama_detection_uvh26 \
        --samples 6
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np


def frame_index(path: Path) -> int:
    """Frame index from a filename. Accepts numeric stems ("000000")
    and prefixed ones ("f_00001", "sample_0005") by taking the
    trailing run of digits."""
    m = re.search(r"(\d+)$", path.stem)
    if not m:
        raise ValueError(f"no trailing digits in filename: {path.name}")
    return int(m.group(1))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--frames_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for reported detections",
    )
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--samples",
        type=int,
        default=6,
        help="Number of annotated sample frames to save",
    )
    ap.add_argument("--seed", type=int, default=42)
    return ap.parse_args(argv)


def draw_boxes(image, boxes, names):
    """Draw boxes with class labels. boxes: (N, 6) = xyxy, conf, cls."""
    out = image.copy()
    for x1, y1, x2, y2, conf, cls in boxes:
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cls_i = int(cls)
        color = (0, 255, 0) if cls_i != 13 else (0, 128, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls_i, cls_i)} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 4)), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + 2, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return out


def main(argv=None) -> int:
    args = parse_args(argv)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("ultralytics not installed", file=sys.stderr)
        return 2

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        print(f"frames dir not found: {frames_dir}", file=sys.stderr)
        return 2

    model = YOLO(args.model)
    names = model.names

    frame_paths = sorted(
        list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.png"))
    )
    print(f"model: {args.model}")
    print(f"classes: {len(names)}")
    print(f"frames: {len(frame_paths)}")
    print(f"confidence threshold: {args.conf}")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    samples_dir = outdir / "samples"
    samples_dir.mkdir(exist_ok=True)

    per_frame: list[dict] = []
    class_counter: Counter = Counter()
    confidences: list[float] = []

    for fp in frame_paths:
        results = model.predict(
            source=str(fp),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
            device="cpu",
        )
        r = results[0]
        boxes = r.boxes
        dets = []
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            conf = boxes.conf.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls, strict=True):
                cname = names.get(int(k), str(k))
                dets.append(
                    {
                        "bbox": [float(x1), float(y1), float(x2), float(y2)],
                        "conf": float(c),
                        "cls": int(k),
                        "name": cname,
                    }
                )
                class_counter[cname] += 1
                confidences.append(float(c))

        per_frame.append(
            {
                "frame": frame_index(fp),
                "num_detections": len(dets),
                "detections": dets,
            }
        )

    counts = [p["num_detections"] for p in per_frame]
    summary = {
        "model": args.model,
        "conf_threshold": args.conf,
        "imgsz": args.imgsz,
        "num_frames": len(per_frame),
        "total_detections": int(sum(counts)),
        "mean_detections_per_frame": (float(np.mean(counts)) if counts else 0.0),
        "std_detections_per_frame": (float(np.std(counts)) if counts else 0.0),
        "min_detections": int(np.min(counts)) if counts else 0,
        "max_detections": int(np.max(counts)) if counts else 0,
        "class_distribution": dict(class_counter.most_common()),
        "mean_confidence": (float(np.mean(confidences)) if confidences else 0.0),
        "std_confidence": (float(np.std(confidences)) if confidences else 0.0),
    }

    (outdir / "detections.json").write_text(json.dumps(per_frame, indent=2) + "\n")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {outdir / 'detections.json'}")
    print(f"wrote {outdir / 'summary.json'}")

    if args.samples > 0 and frame_paths:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(
            len(frame_paths),
            size=min(args.samples, len(frame_paths)),
            replace=False,
        )
        for k, i in enumerate(sorted(idx)):
            fp = frame_paths[i]
            img = cv2.imread(str(fp))
            boxes_arr = np.asarray(
                [[*d["bbox"], d["conf"], d["cls"]] for d in per_frame[i]["detections"]],
                dtype=float,
            ).reshape(-1, 6)
            annotated = draw_boxes(img, boxes_arr, names)
            out_png = samples_dir / f"sample_{k:02d}_frame_{fp.stem}.png"
            cv2.imwrite(str(out_png), annotated)
        print(f"wrote {len(idx)} samples to {samples_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
