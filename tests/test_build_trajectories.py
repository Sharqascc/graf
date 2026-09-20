"""Tests for scripts/build_trajectories.py."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import build_trajectories as bt

REPO = Path(__file__).resolve().parents[1]


def _write_tracks(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return path


def _write_homography(path: Path, H: list[list[float]]) -> Path:
    path.write_text(yaml.safe_dump({"H": H}))
    return path


def _sample_rows() -> list[dict]:
    # One track, two frames; bbox bottom-centers at (10, 20) and (12, 20).
    return [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [5.0, 15.0, 15.0, 20.0],
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": 1,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [7.0, 15.0, 17.0, 20.0],
        },
    ]


# --- loading -------------------------------------------------------


def test_load_tracks_reads_jsonl(tmp_path):
    p = _write_tracks(tmp_path / "t.jsonl", _sample_rows())
    rows = bt.load_tracks(p)
    assert len(rows) == 2
    assert rows[0]["track_id"] == 1


def test_load_tracks_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bt.load_tracks(tmp_path / "nope.jsonl")


def test_load_homography_valid(tmp_path):
    p = _write_homography(tmp_path / "h.yaml", np.eye(3).tolist())
    H = bt.load_homography(p)
    assert H.shape == (3, 3)


def test_load_homography_missing_key_raises(tmp_path):
    p = tmp_path / "h.yaml"
    p.write_text("not_h: 1\n")
    with pytest.raises(ValueError, match="missing 'H' key"):
        bt.load_homography(p)


def test_load_homography_wrong_shape_raises(tmp_path):
    p = _write_homography(tmp_path / "h.yaml", [[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError, match="3x3"):
        bt.load_homography(p)


# --- bbox bottom-center -------------------------------------------


def test_bbox_bottom_center_matches_definition():
    assert bt.bbox_bottom_center([10.0, 20.0, 30.0, 40.0]) == (20.0, 40.0)


# --- projection modes ---------------------------------------------


def test_pixel_mode_passthrough():
    world, source = bt.project_points_to_world(
        [(10.0, 20.0), (30.0, 40.0)],
        homography=None,
        pixels_per_meter=None,
    )
    assert world == [(10.0, 20.0), (30.0, 40.0)]
    assert source == "pixel"


def test_scale_mode_divides_by_ppm():
    world, source = bt.project_points_to_world(
        [(20.0, 40.0), (10.0, 0.0)],
        homography=None,
        pixels_per_meter=2.0,
    )
    assert world == [(10.0, 20.0), (5.0, 0.0)]
    assert source == "scale"


def test_scale_mode_negative_ppm_raises():
    with pytest.raises(ValueError, match="> 0"):
        bt.project_points_to_world([(0.0, 0.0)], homography=None, pixels_per_meter=-1.0)


def test_homography_mode_identity():
    world, source = bt.project_points_to_world(
        [(10.0, 20.0)],
        homography=np.eye(3),
        pixels_per_meter=None,
    )
    assert world == [(10.0, 20.0)]
    assert source == "homography"


def test_homography_wins_over_scale():
    world, source = bt.project_points_to_world(
        [(10.0, 20.0)],
        homography=np.eye(3),
        pixels_per_meter=2.0,
    )
    assert source == "homography"
    assert world == [(10.0, 20.0)]


# --- build_trajectories --------------------------------------------


def test_build_groups_by_track_id():
    payload = bt.build_trajectories(
        _sample_rows(), homography=None, pixels_per_meter=None
    )
    assert len(payload) == 1
    track = payload[0]
    assert track["track_id"] == 1
    assert track["class_name"] == "car"
    assert len(track["frames"]) == 2
    assert track["frames"][0]["frame_id"] == 0
    assert track["frames"][0]["x"] == 10.0
    assert track["frames"][0]["y"] == 20.0


def test_build_multiple_tracks_sorted():
    rows = [
        *_sample_rows(),
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 2,
            "class_name": "pedestrian",
            "confidence": 0.8,
            "bbox_xyxy": [0.0, 0.0, 4.0, 8.0],
        },
    ]
    payload = bt.build_trajectories(rows, homography=None, pixels_per_meter=None)
    assert [str(t["track_id"]) for t in payload] == ["1", "2"]


def test_build_unsorted_frames_are_sorted():
    rows = list(reversed(_sample_rows()))
    payload = bt.build_trajectories(rows, homography=None, pixels_per_meter=None)
    frames = payload[0]["frames"]
    assert [f["frame_id"] for f in frames] == [0, 1]


def test_build_missing_column_raises():
    bad = [{"video_id": "v", "frame_idx": 0, "track_id": 1}]
    with pytest.raises(ValueError, match="missing column 'bbox_xyxy'"):
        bt.build_trajectories(bad, homography=None, pixels_per_meter=None)


def test_build_empty_returns_empty():
    assert bt.build_trajectories([], homography=None, pixels_per_meter=None) == []


# --- end to end ----------------------------------------------------


def test_main_writes_trajectories(tmp_path):
    tracks = _write_tracks(tmp_path / "t.jsonl", _sample_rows())
    out = tmp_path / "out"
    rc = bt.main(
        [
            "--tracks",
            str(tracks),
            "--output_dir",
            str(out),
            "--pixels_per_meter",
            "2.0",
        ]
    )
    assert rc == 0
    payload = json.loads((out / "trajectories.json").read_text())
    assert len(payload) == 1
    assert payload[0]["source"] == "scale"
    assert payload[0]["frames"][0]["x"] == 5.0


def test_main_with_homography(tmp_path):
    tracks = _write_tracks(tmp_path / "t.jsonl", _sample_rows())
    h = _write_homography(tmp_path / "h.yaml", np.eye(3).tolist())
    out = tmp_path / "out"
    rc = bt.main(
        [
            "--tracks",
            str(tracks),
            "--output_dir",
            str(out),
            "--homography_config",
            str(h),
        ]
    )
    assert rc == 0
    payload = json.loads((out / "trajectories.json").read_text())
    assert payload[0]["source"] == "homography"


def test_output_format_compatible_with_compute_ssm(tmp_path):
    """The nested format must be what compute_ssm._flatten_nested accepts."""
    from scripts.compute_ssm import _flatten_nested

    tracks = _write_tracks(tmp_path / "t.jsonl", _sample_rows())
    out = tmp_path / "out"
    bt.main(["--tracks", str(tracks), "--output_dir", str(out)])
    payload = json.loads((out / "trajectories.json").read_text())

    df = _flatten_nested(payload)
    assert len(df) == 2
    assert set(df.columns) >= {
        "video_id",
        "frame_idx",
        "track_id",
        "x_m",
        "y_m",
    }


def test_main_missing_tracks_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        bt.main(
            [
                "--tracks",
                str(tmp_path / "nope.jsonl"),
                "--output_dir",
                str(tmp_path / "out"),
            ]
        )


def test_cli_help():
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "build_trajectories.py"),
            "--help",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--tracks",
        "--output_dir",
        "--homography_config",
        "--pixels_per_meter",
        "--fps",
    ):
        assert flag in r.stdout
