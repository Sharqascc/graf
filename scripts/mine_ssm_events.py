"""Mine discrete SSM events from per-frame metric values.

Reads a per-frame SSM values file (JSON array or JSONL) with at least the
columns ``video_id, frame_idx, track_id_a, track_id_b, metric_name,
value``, calls :func:`graf.ssm.event_mining.mine_events`, and writes the
resulting :class:`~graf.data.schema.SSMEventRecord` rows as JSONL.

Thresholds are loaded from ``configs/ssm/*.yaml`` by default; each config
declares ``metric:`` plus one of ``threshold_seconds`` / ``threshold_mps2``
/ ``threshold``. ``--thresholds-json`` overrides the config discovery for
ad-hoc runs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from graf.ssm.event_mining import load_ssm_thresholds, mine_events
from graf.utils.io import ensure_dir, write_json, write_jsonl

_REQUIRED_COLUMNS = (
    "video_id",
    "frame_idx",
    "track_id_a",
    "track_id_b",
    "metric_name",
    "value",
)


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mine SSM events from per-frame metric values."
    )
    parser.add_argument(
        "--values-path",
        type=str,
        required=True,
        help="JSON array or JSONL file of per-frame SSM values",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/ssm_events",
        help="Output directory",
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/ssm",
        help="Directory of YAML threshold configs",
    )
    parser.add_argument(
        "--thresholds-json",
        type=str,
        default=None,
        help="Inline JSON dict of thresholds, e.g. '{\"TTC\": 1.5}'",
    )
    parser.add_argument(
        "--min-duration-frames",
        type=int,
        default=2,
        help="Minimum run length for an event",
    )
    parser.add_argument(
        "--max-frame-gap",
        type=int,
        default=1,
        help="Maximum frame_idx gap within a run",
    )
    return parser.parse_args(argv)


def load_ssm_values(path: str | Path) -> pd.DataFrame:
    """Load per-frame SSM values from a JSON array or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SSM values file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return pd.DataFrame(columns=list(_REQUIRED_COLUMNS))

    # Try JSON array first; if that fails, treat as JSONL.
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        return pd.DataFrame(rows)

    if isinstance(payload, list):
        return pd.DataFrame(payload)
    if isinstance(payload, dict) and "rows" in payload:
        return pd.DataFrame(payload["rows"])
    raise ValueError(
        f"Unrecognised SSM values format in {path}: expected JSON array, "
        "JSONL, or a dict with a 'rows' key."
    )


def resolve_thresholds(
    config_dir: str | Path,
    thresholds_json: str | None,
) -> dict[str, float]:
    """Prefer --thresholds-json if provided, else load from config dir."""
    if thresholds_json:
        parsed = json.loads(thresholds_json)
        if not isinstance(parsed, dict):
            raise ValueError("--thresholds-json must decode to a JSON object")
        return {str(k): float(v) for k, v in parsed.items()}
    return load_ssm_thresholds(config_dir)


def main(argv=None) -> int:
    args = parse_args(argv)
    outdir = ensure_dir(args.outdir)

    values = load_ssm_values(args.values_path)
    thresholds = resolve_thresholds(args.config_dir, args.thresholds_json)

    if not thresholds:
        print("No thresholds configured — nothing to mine.")
        write_json(outdir / "summary.json", {"num_events": 0, "thresholds": {}})
        return 0

    events = mine_events(
        values,
        thresholds=thresholds,
        min_duration_frames=args.min_duration_frames,
        max_frame_gap=args.max_frame_gap,
    )

    rows = [e.to_dict() for e in events]
    write_jsonl(outdir / "ssm_events.jsonl", rows)
    write_json(
        outdir / "summary.json",
        {
            "num_events": len(events),
            "thresholds": thresholds,
            "input": str(Path(args.values_path)),
            "num_input_rows": int(len(values)),
            "min_duration_frames": args.min_duration_frames,
            "max_frame_gap": args.max_frame_gap,
        },
    )
    print(f"Wrote {len(events)} SSM events to {outdir / 'ssm_events.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
