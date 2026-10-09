"""Tests for scripts/track_quality_summary.py.

The script is a *script*, not a package module, so it is loaded by
path via importlib. Tests cover the pure ``summarise`` function
(empty inputs, single long track, discontinuity detection, class
counts) and the CLI entry point (writes output file, handles missing
inputs with a nonzero exit).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "track_quality_summary_under_test",
    _REPO / "scripts" / "track_quality_summary.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
tqs = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(tqs)


def _det(frame_idx: int, class_name: str = "car") -> dict:
    return {
        "video_id": "v",
        "frame_idx": frame_idx,
        "track_id": 0,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
    }


def _trk(frame_idx: int, track_id: int, class_name: str = "car") -> dict:
    return {
        "video_id": "v",
        "frame_idx": frame_idx,
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox_xyxy": [0.0, 0.0, 10.0, 10.0],
    }


def test_summarise_empty_inputs():
    s = tqs.summarise([], [])
    assert s["num_detections"] == 0
    assert s["num_unique_tracks"] == 0
    assert s["track_length_max"] == 0
    assert s["pct_discontinuous"] == 0.0


def test_summarise_single_long_track():
    tracks = [_trk(f, 1) for f in range(100)]
    s = tqs.summarise([], tracks)
    assert s["num_unique_tracks"] == 1
    assert s["track_length_max"] == 100
    assert s["tracks_ge_50_frames"] == 1
    assert s["tracks_ge_20_frames"] == 1
    assert s["tracks_ge_5_frames"] == 1
    assert s["discontinuous_tracks_gap_gt_3"] == 0


def test_summarise_detects_discontinuity():
    frames = list(range(0, 11)) + [20, 21, 22]
    tracks = [_trk(f, 1) for f in frames]
    s = tqs.summarise([], tracks)
    assert s["discontinuous_tracks_gap_gt_3"] == 1


def test_summarise_counts_tracks_by_class():
    tracks = (
        [_trk(f, 1, "car") for f in range(5)]
        + [_trk(f, 2, "two_wheeler") for f in range(5)]
        + [_trk(f, 3, "two_wheeler") for f in range(5)]
    )
    s = tqs.summarise([], tracks)
    assert s["tracks_by_class"]["car"] == 1
    assert s["tracks_by_class"]["two_wheeler"] == 2


def test_summarise_counts_detections_by_class():
    dets = [_det(i, "two_wheeler") for i in range(7)] + [
        _det(i, "bus") for i in range(2)
    ]
    s = tqs.summarise(dets, [])
    assert s["num_detections"] == 9
    assert s["detections_by_class"]["two_wheeler"] == 7
    assert s["detections_by_class"]["bus"] == 2


def test_main_writes_summary(tmp_path):
    dets = tmp_path / "d.jsonl"
    trks = tmp_path / "t.jsonl"
    out = tmp_path / "summary.json"
    dets.write_text("\n".join(json.dumps(_det(i)) for i in range(3)) + "\n")
    trks.write_text("\n".join(json.dumps(_trk(i, 1)) for i in range(3)) + "\n")

    rc = tqs.main(
        [
            "--detections",
            str(dets),
            "--tracks",
            str(trks),
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["num_detections"] == 3
    assert data["num_unique_tracks"] == 1


def test_main_missing_inputs_returns_1(tmp_path):
    rc = tqs.main(
        [
            "--detections",
            str(tmp_path / "missing.jsonl"),
            "--tracks",
            str(tmp_path / "also_missing.jsonl"),
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    assert rc == 1
