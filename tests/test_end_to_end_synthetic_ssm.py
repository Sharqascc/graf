"""End-to-end test: synthetic trajectories -> per-frame SSM -> mined events.

Exercises scripts/compute_ssm.py and scripts/mine_ssm_events.py on a
synthetic scene where two cars drive head-on. The closing geometry
guarantees TTC drops below the default 3 s threshold, so at least one
TTC event must be mined. DRAC values must be finite throughout.

Runtime target: < 20 s on CPU. No external data.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

FPS = 25.0
DT = 1.0 / FPS


def _write_synthetic_trajectories(path: Path, num_frames: int = 100) -> None:
    """Two cars approaching head-on along y=0, plus a distant pedestrian.

    Cars start at x=-6 and x=+6, each at 2 m/s toward the origin, so
    they come within the 1.5 m collision radius around frame 66. The
    pedestrian at (30, 30) never interacts.
    """
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
                            "video_id": "synthetic_ssm",
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


def _load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def ssm_outputs(tmp_path_factory):
    root = tmp_path_factory.mktemp("ssm_e2e")
    traj = root / "trajectories.jsonl"
    _write_synthetic_trajectories(traj, num_frames=100)

    ssm_dir = root / "ssm_values"
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "compute_ssm.py"),
        "--traj-path",
        str(traj),
        "--outdir",
        str(ssm_dir),
        "--video-id",
        "synthetic_ssm",
        "--fps",
        str(FPS),
        "--distance-threshold",
        "15.0",
        "--collision-radius",
        "1.5",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, (
        f"compute_ssm.py failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )
    return {"root": root, "ssm_dir": ssm_dir}


def test_ssm_values_written_and_well_formed(ssm_outputs):
    values_path = ssm_outputs["ssm_dir"] / "ssm_values.jsonl"
    assert values_path.exists(), f"missing {values_path}"

    rows = _load_jsonl(values_path)
    assert rows, "compute_ssm produced no rows"

    required = {
        "video_id",
        "frame_idx",
        "track_id_a",
        "track_id_b",
        "metric_name",
        "value",
    }
    for r in rows[:5]:
        assert required.issubset(r.keys()), f"missing cols in {r}"

    metrics = {r["metric_name"] for r in rows}
    assert "TTC" in metrics, f"expected TTC rows, got metrics={metrics}"
    assert "DRAC" in metrics, f"expected DRAC rows, got metrics={metrics}"


def test_ttc_drops_below_threshold(ssm_outputs):
    rows = _load_jsonl(ssm_outputs["ssm_dir"] / "ssm_values.jsonl")
    ttc = [r["value"] for r in rows if r["metric_name"] == "TTC"]
    assert ttc, "no TTC rows"
    assert min(ttc) < 3.0, f"head-on approach should produce TTC < 3 s, min={min(ttc)}"


def test_drac_values_are_finite(ssm_outputs):
    rows = _load_jsonl(ssm_outputs["ssm_dir"] / "ssm_values.jsonl")
    drac = [r["value"] for r in rows if r["metric_name"] == "DRAC"]
    assert drac, "no DRAC rows"
    assert all(math.isfinite(v) for v in drac), "DRAC has non-finite values"


def test_mined_events_file_is_well_formed(ssm_outputs):
    values_path = ssm_outputs["ssm_dir"] / "ssm_values.jsonl"
    events_dir = ssm_outputs["root"] / "ssm_events"

    cmd = [
        sys.executable,
        str(REPO / "scripts" / "mine_ssm_events.py"),
        "--values-path",
        str(values_path),
        "--outdir",
        str(events_dir),
        "--config-dir",
        str(REPO / "configs" / "ssm"),
        "--min-duration-frames",
        "2",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO)
    assert res.returncode == 0, (
        f"mine_ssm_events.py failed:\nstdout:\n{res.stdout}\nstderr:\n{res.stderr}"
    )

    event_files = sorted(events_dir.glob("*.jsonl"))
    assert event_files, f"no *.jsonl produced in {events_dir}"

    events = _load_jsonl(event_files[0])
    assert events, f"{event_files[0]} is empty"

    e = events[0]
    for key in (
        "video_id",
        "event_id",
        "metric_name",
        "track_id_a",
        "track_id_b",
        "start_frame",
        "end_frame",
        "min_value",
        "threshold",
        "severity",
    ):
        assert key in e, f"missing key {key} in {e}"

    ttc_events = [e for e in events if e["metric_name"] == "TTC"]
    assert ttc_events, (
        f"expected at least one TTC event, got "
        f"metrics={[e['metric_name'] for e in events]}"
    )
