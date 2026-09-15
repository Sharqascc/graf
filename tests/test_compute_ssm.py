"""Tests for scripts/compute_ssm.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import compute_ssm as script_mod

REPO = Path(__file__).resolve().parents[1]


def _write_nested(path: Path) -> Path:
    """Two cars, head-on, converging. 3 frames each."""
    payload = [
        {
            "track_id": 1,
            "class_name": "car",
            "frames": [
                {"frame_id": 0, "x": -5.0, "y": 0.0},
                {"frame_id": 1, "x": -3.0, "y": 0.0},
                {"frame_id": 2, "x": -1.0, "y": 0.0},
            ],
        },
        {
            "track_id": 2,
            "class_name": "car",
            "frames": [
                {"frame_id": 0, "x": 5.0, "y": 0.0},
                {"frame_id": 1, "x": 3.0, "y": 0.0},
                {"frame_id": 2, "x": 1.0, "y": 0.0},
            ],
        },
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_flat_jsonl(path: Path) -> Path:
    rows = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": "a",
            "x_m": 0.0,
            "y_m": 0.0,
            "vx_mps": 1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": "b",
            "x_m": 6.0,
            "y_m": 0.0,
            "vx_mps": -1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": "a",
            "x_m": 1.0,
            "y_m": 0.0,
            "vx_mps": 1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": "b",
            "x_m": 5.0,
            "y_m": 0.0,
            "vx_mps": -1.0,
            "vy_mps": 0.0,
        },
    ]
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def test_load_trajectories_nested(tmp_path):
    p = _write_nested(tmp_path / "t.json")
    df = script_mod.load_trajectories(p)
    assert len(df) == 6
    assert set(df.columns) >= {"frame_idx", "track_id", "x_m", "y_m"}


def test_load_trajectories_flat_jsonl(tmp_path):
    p = _write_flat_jsonl(tmp_path / "t.jsonl")
    df = script_mod.load_trajectories(p)
    assert len(df) == 4
    assert "vx_mps" in df.columns


def test_load_trajectories_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.load_trajectories(tmp_path / "nope.json")


def test_compute_values_head_on_produces_ttc_and_drac(tmp_path):
    p = _write_nested(tmp_path / "t.json")
    df = script_mod._normalize_flat(script_mod.load_trajectories(p), video_id="v")
    rows = script_mod.compute_values(
        df,
        video_id="v",
        fps=10.0,
        distance_threshold=15.0,
        collision_radius=1.5,
    )
    metrics = {r["metric_name"] for r in rows}
    assert metrics == {"TTC", "DRAC"}
    # All rows are finite values
    for r in rows:
        assert isinstance(r["value"], float)
        assert r["value"] == r["value"]  # not NaN
        assert r["value"] != float("inf")


def test_compute_values_diverging_yields_no_ttc(tmp_path):
    # Two actors moving away from each other, never closing.
    rows = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": "a",
            "x_m": 0.0,
            "y_m": 0.0,
            "vx_mps": -1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": "b",
            "x_m": 5.0,
            "y_m": 0.0,
            "vx_mps": 1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": "a",
            "x_m": -1.0,
            "y_m": 0.0,
            "vx_mps": -1.0,
            "vy_mps": 0.0,
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": "b",
            "x_m": 6.0,
            "y_m": 0.0,
            "vx_mps": 1.0,
            "vy_mps": 0.0,
        },
    ]
    import pandas as pd

    df = script_mod._normalize_flat(pd.DataFrame(rows), video_id="v")
    values = script_mod.compute_values(
        df,
        video_id="v",
        fps=10.0,
        distance_threshold=15.0,
        collision_radius=1.5,
    )
    # TTC is +inf for diverging actors, so no TTC rows; DRAC is 0.0 which
    # is finite and emitted.
    ttc_rows = [r for r in values if r["metric_name"] == "TTC"]
    drac_rows = [r for r in values if r["metric_name"] == "DRAC"]
    assert ttc_rows == []
    assert len(drac_rows) > 0
    assert all(r["value"] == 0.0 for r in drac_rows)


def test_main_writes_values_file(tmp_path):
    p = _write_nested(tmp_path / "t.json")
    outdir = tmp_path / "out"
    rc = script_mod.main(
        [
            "--traj-path",
            str(p),
            "--outdir",
            str(outdir),
            "--fps",
            "10.0",
            "--video-id",
            "v",
        ]
    )
    assert rc == 0
    values_path = outdir / "ssm_values.jsonl"
    summary_path = outdir / "summary.json"
    assert values_path.exists()
    assert summary_path.exists()
    rows = [json.loads(line) for line in values_path.read_text().splitlines()]
    assert len(rows) > 0
    assert all("video_id" in r and r["video_id"] == "v" for r in rows)
    summary = json.loads(summary_path.read_text())
    assert summary["num_rows"] == len(rows)


def test_main_with_mine_events(tmp_path):
    p = _write_nested(tmp_path / "t.json")
    values_dir = tmp_path / "values"
    events_dir = tmp_path / "events"
    rc = script_mod.main(
        [
            "--traj-path",
            str(p),
            "--outdir",
            str(values_dir),
            "--fps",
            "10.0",
            "--video-id",
            "v",
            "--mine-events",
            "--mine-config-dir",
            str(REPO / "configs" / "ssm"),
            "--mine-outdir",
            str(events_dir),
        ]
    )
    assert rc == 0
    assert (values_dir / "ssm_values.jsonl").exists()
    events_path = events_dir / "ssm_events.jsonl"
    assert events_path.exists()
    # Head-on converging cars: should produce at least one event
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert len(events) >= 1
    for e in events:
        assert e["metric_name"] in {"TTC", "DRAC", "PET"}


def test_main_missing_input_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.main(
            [
                "--traj-path",
                str(tmp_path / "missing.json"),
                "--outdir",
                str(tmp_path / "out"),
            ]
        )


def test_script_cli_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "compute_ssm.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--traj-path",
        "--outdir",
        "--fps",
        "--distance-threshold",
        "--collision-radius",
        "--mine-events",
        "--mine-config-dir",
        "--mine-outdir",
        "--video-id",
    ):
        assert flag in r.stdout
