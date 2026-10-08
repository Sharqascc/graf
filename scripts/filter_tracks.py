"""Filter a tracks JSONL to tracks with at least N frames.

A tracks JSONL is one row per detection. Track length is the number of
rows with a given track_id. This script writes a filtered JSONL
containing only rows whose track_id has at least --min-frames rows.

Usage:
    python scripts/filter_tracks.py --input <tracks.jsonl> \
        --output <filtered.jsonl> --min-frames 20
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-frames", type=int, default=20)
    args = ap.parse_args(argv)

    counts: dict[int, int] = defaultdict(int)
    rows = []
    for line in Path(args.input).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        counts[r["track_id"]] += 1
        rows.append(r)

    keep = {tid for tid, n in counts.items() if n >= args.min_frames}
    with Path(args.output).open("w") as f:
        for r in rows:
            if r["track_id"] in keep:
                f.write(json.dumps(r) + "\n")

    print(f"filtered to {len(keep)} tracks (from {len(counts)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
