"""Run YOLOv8 or UVH-26 detection on a directory of frame images.

Reads a YAML config (default: ``configs/detection/yolov8.yaml``) with keys:

    weights          path or model name passed to ultralytics.YOLO()
    conf_threshold   minimum detection confidence
    iou_threshold    NMS IoU threshold
    imgsz            inference image size (square)
    classes          list of native class names to keep (null = all mapped)
    class_map        mapping from detector-native class names to GRAF actor
                     classes. Defaults to COCO_TO_GRAF when absent.

Any of ``--model``, ``--imgsz``, ``--conf``, ``--iou`` override the
corresponding config value. Detections whose COCO class does not map to a
GRAF actor class are dropped, as are classes excluded by ``classes`` if
that key is set.

Writes one JSON object per line to ``<output_dir>/detections.jsonl``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml

COCO_TO_GRAF = {
    "car": "car",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "two_wheeler",
}

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[1] / "configs" / "detection" / "yolov8.yaml"
)


def load_config(path: str | Path) -> dict:
    """Load a detection config YAML into a dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Detection config not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping, got {type(data).__name__}")
    return data


def frame_index(path: Path) -> int:
    """Frame index from a filename.

    Accepts numeric stems ("000000") and prefixed ones ("f_00001",
    "frame_0005") by taking the trailing run of digits. Falls back to
    -1 if the stem has no trailing digits, so the caller can decide.
    """
    m = re.search(r"(\d+)$", path.stem)
    if not m:
        return -1
    return int(m.group(1))


def resolve(cfg: dict, key: str, cli_value, default):
    """CLI value wins; else config; else default."""
    if cli_value is not None:
        return cli_value
    value = cfg.get(key)
    return default if value is None else value


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run YOLOv8 detection on a directory of frames."
    )
    p.add_argument(
        "--frames_dir", required=True, help="Directory of *.jpg frame images"
    )
    p.add_argument("--output_dir", required=True, help="Directory for detections.jsonl")
    p.add_argument(
        "--config", default=str(_DEFAULT_CONFIG), help="Detection config YAML"
    )
    p.add_argument("--model", default=None, help="Override config 'weights'")
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument(
        "--video-id",
        default="video",
        help="Identifier written into each detection record",
    )
    p.add_argument("--imgsz", type=int, default=None, help="Override config 'imgsz'")
    p.add_argument(
        "--conf", type=float, default=None, help="Override config 'conf_threshold'"
    )
    p.add_argument(
        "--iou", type=float, default=None, help="Override config 'iou_threshold'"
    )
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)

    weights = resolve(cfg, "weights", args.model, "yolov8n.pt")
    imgsz = resolve(cfg, "imgsz", args.imgsz, 640)
    conf = resolve(cfg, "conf_threshold", args.conf, 0.25)
    iou = resolve(cfg, "iou_threshold", args.iou, 0.5)
    classes_filter = cfg.get("classes")  # None -> keep all mapped
    class_map = cfg.get("class_map") or COCO_TO_GRAF

    # Lazy import — ultralytics is an optional 'detection' extra.
    from ultralytics import YOLO

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(frames_dir.glob("*.jpg")) + sorted(frames_dir.glob("*.png"))
    model = YOLO(weights)
    out_path = output_dir / "detections.jsonl"

    written = 0
    with out_path.open("w") as f:
        for i, frame_path in enumerate(frame_paths):
            if i % args.stride != 0:
                continue
            frame_idx = frame_index(frame_path)
            if frame_idx < 0:
                frame_idx = i
            results = model.predict(
                str(frame_path),
                imgsz=imgsz,
                conf=conf,
                iou=iou,
                verbose=False,
            )[0]
            boxes = results.boxes
            if boxes is None:
                continue
            for box in boxes:
                native_name = results.names[int(box.cls[0])]
                if native_name not in class_map:
                    continue
                if classes_filter and native_name not in classes_filter:
                    continue
                record = {
                    "video_id": args.video_id,
                    "frame_idx": frame_idx,
                    "actor_id": None,
                    "class_name": class_map[native_name],
                    "confidence": float(box.conf[0]),
                    "bbox_xyxy": box.xyxy[0].tolist(),
                }
                f.write(json.dumps(record) + "\n")
                written += 1

    print(f"Wrote {written} detections to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
