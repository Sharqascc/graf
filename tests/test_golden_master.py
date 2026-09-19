"""Golden-master regression test for the SSM pipeline.

Runs compute_ssm.py + mine_ssm_events.py on a fixed synthetic scene and
compares the summary against tests/snapshots/golden_ssm_v1.json. Any
change to TTC/DRAC arithmetic, event-mining thresholds, or the on-disk
formats that shifts these values beyond rel_tol=1e-6 fails the test.

Regenerate after an intentional change:

    REGEN_SNAPSHOT=1 python -m pytest tests/test_golden_master.py

Runtime target: < 15 s on CPU. No external data.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SNAPSHOT = Path(__file__).parent / "snapshots" / "golden_ssm_v1.json"

FPS = 25.0
DT = 1.0 / FPS
REGEN = os.environ.get("REGEN_SNAPSHOT") == "1"


def _write_synthetic_trajectories(path: Path, num_frames: int = 100) -> None:
    actors = [
        {"track_id": 101, "x": -6.0, "y": 0.0, "vx": 2.0, "vy": 0.0},
        {"track_id": 102, "x": 6.0, "y": 0.0, "vx": -2.0, "vy": 0.0},
        {"track_id": 201, "x": 30.0, "y": 30.0, "vx": 0.0, "vy": 0.0},
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for frame_idx in range(num_frames):
            for a in actors:
                f.write(
                    json.dumps(
                        {
                            "video_id": "golden_ssm",
                            "frame_idx": frame_idx,
                            "track_id": a["track_id"],
                            "x_m": a["x"] + a["vx"] * frame_idx * DT,
                            "y_m": a["y"] + a["vy"] * frame_idx * DT,
                            "vx_mps": a["vx"],
                            "vy_mps": a["vy"],
                        }
                    )
                    + "\n"
                )


def _run_pipeline(tmp_path: Path) -> dict:
    traj = tmp_path / "trajectories.jsonl"
    _write_synthetic_trajectories(traj, num_frames=100)

    ssm_dir = tmp_path / "ssm_values"
    events_dir = tmp_path / "ssm_events"

    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "compute_ssm.py"),
            "--traj-path",
            str(traj),
            "--outdir",
            str(ssm_dir),
            "--video-id",
            "golden_ssm",
            "--fps",
            str(FPS),
            "--distance-threshold",
            "15.0",
            "--collision-radius",
            "1.5",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, f"compute_ssm.py failed: {r.stderr}"

    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "mine_ssm_events.py"),
            "--values-path",
            str(ssm_dir / "ssm_values.jsonl"),
            "--outdir",
            str(events_dir),
            "--config-dir",
            str(REPO / "configs" / "ssm"),
            "--min-duration-frames",
            "2",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, f"mine_ssm_events.py failed: {r.stderr}"

    values = [
        json.loads(line)
        for line in (ssm_dir / "ssm_values.jsonl").read_text().splitlines()
        if line.strip()
    ]
    events: list[dict] = []
    for p in sorted(events_dir.glob("*.jsonl")):
        events.extend(
            json.loads(line) for line in p.read_text().splitlines() if line.strip()
        )
    return {"values": values, "events": events}


def _summarize(values: list[dict], events: list[dict]) -> dict:
    ttc = sorted(r["value"] for r in values if r["metric_name"] == "TTC")
    drac = sorted(r["value"] for r in values if r["metric_name"] == "DRAC")
    event_summary = sorted(
        [
            {
                "metric_name": e["metric_name"],
                "start_frame": e["start_frame"],
                "end_frame": e["end_frame"],
                "min_value": round(float(e["min_value"]), 6),
            }
            for e in events
        ],
        key=lambda e: (e["metric_name"], e["start_frame"], e["end_frame"]),
    )
    return {
        "num_value_rows": len(values),
        "num_ttc_rows": len(ttc),
        "num_drac_rows": len(drac),
        "ttc_min": round(ttc[0], 6) if ttc else None,
        "ttc_max": round(ttc[-1], 6) if ttc else None,
        "ttc_mean": round(sum(ttc) / len(ttc), 6) if ttc else None,
        "drac_min": round(drac[0], 6) if drac else None,
        "drac_max": round(drac[-1], 6) if drac else None,
        "drac_mean": round(sum(drac) / len(drac), 6) if drac else None,
        "num_events": len(events),
        "events": event_summary,
    }


@pytest.fixture(scope="module")
def summary(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("golden")
    data = _run_pipeline(tmp)
    return _summarize(data["values"], data["events"])


def test_golden_master_ssm_snapshot(summary):
    if REGEN:
        SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
        SNAPSHOT.write_text(json.dumps(summary, indent=2, sort_keys=True))
        pytest.skip("regenerated snapshot")

    assert SNAPSHOT.exists(), (
        f"snapshot missing: {SNAPSHOT}\n"
        "regenerate with REGEN_SNAPSHOT=1 pytest tests/test_golden_master.py"
    )
    expected = json.loads(SNAPSHOT.read_text())

    for key in ("num_value_rows", "num_ttc_rows", "num_drac_rows", "num_events"):
        assert summary[key] == expected[key], (
            f"{key}: {summary[key]} != {expected[key]}"
        )

    for key in ("ttc_min", "ttc_max", "ttc_mean", "drac_min", "drac_max", "drac_mean"):
        a, b = summary[key], expected[key]
        if a is None or b is None:
            assert a == b, f"{key}: {a} != {b}"
            continue
        assert math.isclose(a, b, rel_tol=1e-6, abs_tol=1e-9), f"{key}: {a} != {b}"

    assert len(summary["events"]) == len(expected["events"]), (
        f"event count: {len(summary['events'])} != {len(expected['events'])}"
    )
    for got, exp in zip(summary["events"], expected["events"]):
        assert got["metric_name"] == exp["metric_name"]
        assert got["start_frame"] == exp["start_frame"]
        assert got["end_frame"] == exp["end_frame"]
        assert math.isclose(
            got["min_value"], exp["min_value"], rel_tol=1e-6, abs_tol=1e-9
        )
