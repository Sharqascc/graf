import argparse
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video file")
    parser.add_argument("--video_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--stride", type=int, default=1, help="Save every Nth frame")
    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Window start in seconds (default: 0)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Window length in seconds (default: to end of video)",
    )
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    save_dir = Path(args.output_dir) / video_path.stem
    save_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0
    start_frame = int(args.start_time * fps)
    end_frame = (
        int((args.start_time + args.duration) * fps)
        if args.duration is not None
        else None
    )

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < start_frame:
            frame_idx += 1
            continue
        if end_frame is not None and frame_idx >= end_frame:
            break
        if (frame_idx - start_frame) % args.stride == 0:
            out_path = save_dir / f"{frame_idx:06d}.jpg"
            cv2.imwrite(str(out_path), frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Extracted {saved} frames to {save_dir}")


if __name__ == "__main__":
    main()
