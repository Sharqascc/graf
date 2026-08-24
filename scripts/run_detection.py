
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

COCO_TO_GRAF = {
    "car": "car",
    "person": "pedestrian",
    "bicycle": "bicycle",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "two_wheeler",
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    frame_paths = sorted(frames_dir.glob("*.jpg"))
    out_path = output_dir / "detections.jsonl"

    with out_path.open("w") as f:
        for frame_idx, frame_path in enumerate(frame_paths):
            if frame_idx % args.stride != 0:
                continue
            results = model.predict(str(frame_path), verbose=False)[0]
            boxes = results.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                class_name = results.names[cls_id]
                if class_name not in COCO_TO_GRAF:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                record = {
                    "video_id": "sample_video",
                    "frame_idx": frame_idx,
                    "actor_id": None,
                    "class_name": COCO_TO_GRAF[class_name],
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2],
                }
                f.write(json.dumps(record) + "\n")

    print(f"Detection saved to {out_path}")

if __name__ == "__main__":
    main()
