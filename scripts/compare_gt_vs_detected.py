"""Compare TTC-window labels from GT tracks vs detected tracks.

The paper's reported labels come from VNTraffic hand-annotated
ground-truth boxes (see docs/paper/method.md section 1.4). This
script answers the deployment question: if the same pipeline ran on
detections instead of GT, how much would the window-level label
vector change?

Steps:
  1. Extract frames from the video.
  2. Run YOLOv8 detection over the frames.
  3. Run the greedy IoU tracker over the detections.
  4. Convert GT MOT text to a tracks JSONL (same shape as
     scripts/prepare_vntraffic.py writes).
  5. Apply identical filtering, world-coordinate projection, and
     sustained-label computation to both track sets.
  6. Report agreement, disagreement direction, and per-window
     diffs.

Usage:
    python scripts/compare_gt_vs_detected.py \\
        --video data/external/vntraffic/VNTraffic/VNTraffic_Original-video.mp4 \\
        --gt-txt data/external/vntraffic/VNTraffic/VNTraffic_GroundTruth.txt \\
        --homography-config data/raw/vntraffic_homography.yaml \\
        --output-dir outputs/gt_vs_detected

This script does not run in CI on every push. It is triggered
manually from the GT vs Detected workflow in .github/workflows/.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO))

from graf.training.conflict_pairs import (  # noqa: E402
    add_world_coords,
    filter_tracks,
    load_tracks,
)
from scripts.label_strategies import label_sustained  # noqa: E402
from scripts.prepare_vntraffic import convert_tracks  # noqa: E402


class _Window:
    """Minimal window object matching what label_sustained reads."""
    __slots__ = ('frame_ids',)

    def __init__(self, frame_ids):
        self.frame_ids = np.asarray(frame_ids, dtype=int)


class _WindowDS:
    __slots__ = ('windows',)

    def __init__(self, windows):
        self.windows = [_Window(w) for w in windows]

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i]


def build_windows(num_frames: int, window_size: int, stride: int) -> _WindowDS:
    """Contiguous overlapping windows over the frame index range."""
    windows = []
    for start in range(0, num_frames - window_size + 1, stride):
        windows.append(list(range(start, start + window_size)))
    return _WindowDS(windows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--video", required=True, help="Original clip .mp4")
    p.add_argument("--gt-txt", required=True, help="MOT-format GT text file")
    p.add_argument("--homography-config", required=True, help="Homography YAML")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--window-size", type=int, default=5)
    p.add_argument("--stride", type=int, default=2)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--run-length", type=int, default=5)
    p.add_argument("--ttc-threshold-seconds", type=float, default=1.5)
    p.add_argument("--ttc-distance-threshold", type=float, default=3.0)
    p.add_argument("--min-track-len", type=int, default=5)
    p.add_argument("--min-confidence", type=float, default=0.4)
    p.add_argument("--detection-imgsz", type=int, default=640)
    p.add_argument("--skip-detection", action="store_true",
                   help="Reuse existing frames/detections/tracks if present")
    return p.parse_args(argv)


def run_step(cmd):
    print(f'$ {" ".join(str(c) for c in cmd)}')
    r = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip()[-800:])
    if r.returncode != 0:
        print(r.stderr.strip()[-1500:], file=sys.stderr)
        raise SystemExit(f'step failed: {cmd[0]}')
    return r


def main(argv=None) -> int:
    args = parse_args(argv)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. GT -> tracks.jsonl
    gt_tracks_path = out / 'gt_tracks.jsonl'
    if not gt_tracks_path.exists():
        stats = convert_tracks(Path(args.gt_txt), gt_tracks_path)
        print(f'GT tracks: {stats}')

    # 2. Detection + tracking on the video
    frames_root = out / 'frames'
    detections_dir = out / 'detections'
    tracks_dir = out / 'tracks'
    detected_tracks_path = tracks_dir / 'tracks.jsonl'

    if not (args.skip_detection and detected_tracks_path.exists()):
        if not frames_root.exists():
            run_step([
                sys.executable, 'scripts/extract_frames.py',
                '--video_path', args.video,
                '--output_dir', str(frames_root),
            ])
        video_stem = Path(args.video).stem
        actual_frames = frames_root / video_stem

        run_step([
            sys.executable, 'scripts/run_detection.py',
            '--frames_dir', str(actual_frames),
            '--output_dir', str(detections_dir),
            '--imgsz', str(args.detection_imgsz),
        ])

        run_step([
            sys.executable, 'scripts/run_tracking.py',
            '--detections', str(detections_dir / 'detections.jsonl'),
            '--output_dir', str(tracks_dir),
        ])

    # 3. Load and project both track sets
    with open(args.homography_config) as f:
        H = np.array(yaml.safe_load(f)['H'], dtype=np.float64)

    gt_df = filter_tracks(load_tracks(gt_tracks_path),
                          min_conf=args.min_confidence,
                          min_len=args.min_track_len)
    gt_df = add_world_coords(gt_df, H, fps=args.fps)

    det_df = filter_tracks(load_tracks(detected_tracks_path),
                           min_conf=args.min_confidence,
                           min_len=args.min_track_len)
    det_df = add_world_coords(det_df, H, fps=args.fps)

    # 4. Window structure from max frame index in either set
    num_frames = int(max(gt_df['frame_idx'].max(),
                         det_df['frame_idx'].max())) + 1
    window_ds = build_windows(num_frames, args.window_size, args.stride)

    # 5. Compute labels on both
    common = dict(
        ttc_threshold_seconds=args.ttc_threshold_seconds,
        distance_threshold=args.ttc_distance_threshold,
        run_length=args.run_length,
    )
    gt_labels = label_sustained(gt_df, window_ds, **common)
    det_labels = label_sustained(det_df, window_ds, **common)

    # 6. Compare
    n = len(gt_labels)
    agree = sum(1 for a, b in zip(gt_labels, det_labels) if a == b)
    gt_pos = sum(gt_labels)
    det_pos = sum(det_labels)
    gt_pos_det_neg = sum(1 for a, b in zip(gt_labels, det_labels) if a == 1 and b == 0)
    gt_neg_det_pos = sum(1 for a, b in zip(gt_labels, det_labels) if a == 0 and b == 1)

    result = {
        'num_windows': n,
        'window_size': args.window_size,
        'stride': args.stride,
        'run_length': args.run_length,
        'gt_positives': gt_pos,
        'detected_positives': det_pos,
        'gt_majority': max(gt_pos, n - gt_pos) / n if n else 0.0,
        'detected_majority': max(det_pos, n - det_pos) / n if n else 0.0,
        'agreement': agree,
        'agreement_rate': agree / n if n else 0.0,
        'gt_pos_detected_neg': gt_pos_det_neg,
        'gt_neg_detected_pos': gt_neg_det_pos,
        'gt_track_count': int(gt_df['track_id'].nunique()),
        'detected_track_count': int(det_df['track_id'].nunique()),
        'gt_row_count': int(len(gt_df)),
        'detected_row_count': int(len(det_df)),
        'gt_labels': gt_labels,
        'detected_labels': det_labels,
    }
    (out / 'comparison.json').write_text(json.dumps(result, indent=2))

    md = [
        '# GT vs detected — TTC window-label comparison',
        '',
        f'- Windows: {n} (window_size={args.window_size}, stride={args.stride}, run_length={args.run_length})',
        f'- GT positives:       {gt_pos} ({100 * gt_pos / n:.1f}%)' if n else '- GT positives: 0',
        f'- Detected positives: {det_pos} ({100 * det_pos / n:.1f}%)' if n else '- Detected positives: 0',
        f'- Agreement: {agree}/{n} ({100 * agree / n:.1f}%)' if n else '- Agreement: n/a',
        '',
        '## Disagreement direction',
        '',
        f'- GT=1, Detected=0: {gt_pos_det_neg}',
        f'- GT=0, Detected=1: {gt_neg_det_pos}',
        '',
        '## Track counts (post-filter)',
        '',
        f'- GT tracks:       {result["gt_track_count"]}',
        f'- Detected tracks: {result["detected_track_count"]}',
        f'- GT rows:         {result["gt_row_count"]}',
        f'- Detected rows:   {result["detected_row_count"]}',
    ]
    (out / 'comparison.md').write_text('\n'.join(md) + '\n')

    print()
    print('\n'.join(md))
    print(f'\nWrote {out / "comparison.json"}')
    print(f'Wrote {out / "comparison.md"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
