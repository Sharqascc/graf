from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute toy surrogate safety measures for GRAF."
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        default="outputs/trajectories/trajectories.json",
        help="Input trajectories JSON",
    )
    parser.add_argument(
        "--outdir", type=str, default="outputs/ssm_events", help="Output directory"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    traj_path = Path(args.traj_path)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectories file: {traj_path}")

    trajectories = json.loads(traj_path.read_text(encoding="utf-8"))
    events = []
    for idx, traj in enumerate(trajectories, start=1):
        frames = traj.get("frames", [])
        if len(frames) < 2:
            continue
        event = {
            "event_id": f"ssm_{idx:03d}",
            "track_id": traj.get("track_id"),
            "class_name": traj.get("class_name"),
            "measure": "ttc",
            "value": max(0.0, 5.0 - 0.5 * len(frames)),
            "status": "toy_computed",
        }
        events.append(event)

    out_path = outdir / "ssm_events.json"
    out_path.write_text(json.dumps(events, indent=2), encoding="utf-8")
    print(f"Wrote {len(events)} SSM events to {out_path}")


if __name__ == "__main__":
    main()
