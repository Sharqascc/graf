from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build toy temporal windows for GRAF.")
    parser.add_argument(
        "--graph-dir", type=str, default="outputs/graphs", help="Input graph directory"
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs/windows", help="Output directory"
    )
    parser.add_argument("--window-size", type=int, default=2, help="Graphs per window")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    graph_files = sorted(graph_dir.glob("*.json"))
    if not graph_files:
        raise RuntimeError(f"No graph JSON files found in {graph_dir}")

    windows = []
    for start in range(0, len(graph_files), args.window_size):
        chunk = graph_files[start : start + args.window_size]
        window = {
            "window_id": f"window_{start // args.window_size + 1:03d}",
            "graph_files": [str(p) for p in chunk],
            "num_graphs": len(chunk),
        }
        out_path = outdir / f"{window['window_id']}.json"
        out_path.write_text(json.dumps(window, indent=2), encoding="utf-8")
        windows.append(str(out_path))

    print(f"Wrote {len(windows)} windows to {outdir}")
    for path in windows:
        print(path)


if __name__ == "__main__":
    main()
