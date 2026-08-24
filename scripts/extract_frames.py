
import argparse
from pathlib import Path
import cv2

def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    save_dir = Path(args.output_dir) / video_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % args.stride == 0:
            out_path = save_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames to {save_dir}")

if __name__ == "__main__":
    main()
