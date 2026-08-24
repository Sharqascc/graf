
import argparse
import json
from pathlib import Path
from collections import defaultdict

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    denom = boxAArea + boxBArea - interArea
    if denom <= 0:
        return 0.0
    return interArea / denom

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detections", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iou_threshold", type=float, default=0.3)
    args = parser.parse_args()

    detections = []
    with open(args.detections) as f:
        for line in f:
            if line.strip():
                detections.append(json.loads(line))

    # Group by frame
    by_frame = defaultdict(list)
    for det in detections:
        by_frame[det["frame_idx"]].append(det)

    tracks = []
    active_tracks = {}  # track_id -> last bbox + class + frames_since_update
    next_id = 0

    for frame_idx in sorted(by_frame):
        dets = by_frame[frame_idx]
        unmatched = list(range(len(dets)))
        assigned = set()

        for tid, info in list(active_tracks.items()):
            if unmatched:
                best_iou = 0
                best_det = -1
                for i in unmatched:
                    det = dets[i]
                    if det["class_name"] != info["class_name"]:
                        continue
                    score = iou(info["bbox_xyxy"], det["bbox_xyxy"])
                    if score > best_iou:
                        best_iou = score
                        best_det = i
                if best_det >= 0 and best_iou >= args.iou_threshold:
                    det = dets[best_det]
                    active_tracks[tid] = {
                        "bbox_xyxy": det["bbox_xyxy"],
                        "class_name": det["class_name"],
                        "frames_since_update": 0,
                    }
                    tracks.append({
                        "video_id": det["video_id"],
                        "frame_idx": det["frame_idx"],
                        "track_id": tid,
                        "class_name": det["class_name"],
                        "confidence": det["confidence"],
                        "bbox_xyxy": det["bbox_xyxy"],
                    })
                    assigned.add(best_det)
                    unmatched.remove(best_det)

        # New tracks for unmatched detections
        for i in unmatched:
            det = dets[i]
            tid = next_id
            next_id += 1
            active_tracks[tid] = {
                "bbox_xyxy": det["bbox_xyxy"],
                "class_name": det["class_name"],
                "frames_since_update": 0,
            }
            tracks.append({
                "video_id": det["video_id"],
                "frame_idx": det["frame_idx"],
                "track_id": tid,
                "class_name": det["class_name"],
                "confidence": det["confidence"],
                "bbox_xyxy": det["bbox_xyxy"],
            })

        # Remove tracks not seen for 30 frames
        stale = [tid for tid, info in active_tracks.items() if info["frames_since_update"] > 30]
        for tid in stale:
            del active_tracks[tid]

    out_path = Path(args.output_dir) / "tracks.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for t in tracks:
            f.write(json.dumps(t) + "\n")

    print(f"Tracks saved to {out_path}")

if __name__ == "__main__":
    main()
