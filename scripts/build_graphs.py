
import argparse
import sys
import json
from pathlib import Path
from collections import defaultdict

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from graf.graph.builders import GraphBuilder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--pixels_per_meter", type=float, default=20.0,
                        help="Approximate conversion from pixels to meters")
    parser.add_argument("--radius", type=float, default=15.0)
    args = parser.parse_args()

    # Read all tracks
    tracks = []
    with open(args.tracks) as f:
        for line in f:
            if line.strip():
                tracks.append(json.loads(line))

    # Group by frame
    by_frame = defaultdict(list)
    for t in tracks:
        by_frame[t["frame_idx"]].append(t)

    # Build graph per frame
    builder = GraphBuilder(radius=args.radius)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prev_positions = {}  # track_id -> (x_pix, y_pix) from previous frame
    saved = 0
    for frame_idx in sorted(by_frame):
        records = []
        current_positions = {}
        for t in by_frame[frame_idx]:
            x1, y1, x2, y2 = t["bbox_xyxy"]
            x_pix = (x1 + x2) / 2.0
            y_pix = (y1 + y2) / 2.0
            current_positions[t["track_id"]] = (x_pix, y_pix)

            # Convert to approximate world coordinates (meters)
            x_m = x_pix / args.pixels_per_meter
            y_m = y_pix / args.pixels_per_meter

            # Simple velocity from previous frame for same track
            vx = vy = 0.0
            if t["track_id"] in prev_positions:
                px_prev, py_prev = prev_positions[t["track_id"]]
                vx = (x_pix - px_prev) / args.pixels_per_meter
                vy = (y_pix - py_prev) / args.pixels_per_meter

            records.append({
                "track_id": int(t["track_id"]),
                "actor_class": t["class_name"],
                "x_m": x_m,
                "y_m": y_m,
                "vx": vx,
                "vy": vy,
            })

        data = builder.build_pyg_data(
            records,
            frame_id=int(frame_idx),
            video_id="sample_video",
        )

        # Save graph
        path = out_dir / f"graph_f{int(frame_idx):06d}.pt"
        torch.save(data, path)
        saved += 1
        prev_positions = current_positions

    print(f"Saved {saved} graphs to {out_dir}")

if __name__ == "__main__":
    main()
