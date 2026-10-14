"""Tests for the Phase 1 + Phase 2 gates in scripts/run_tracking.py.

Run from the SCI tracking improvement plan:
docs/sci_tracking_improvement_plan.md. The gates live in scripts/,
so this file loads the script by path and targets the pure functions
and their wiring inside track().
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_tracking_gates_under_test",
    _REPO / "scripts" / "run_tracking.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
rt = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rt
_SPEC.loader.exec_module(rt)

SQUARE = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]


def test_point_in_polygon_inside_and_outside():
    assert rt._point_in_polygon(50.0, 50.0, SQUARE)
    assert not rt._point_in_polygon(150.0, 50.0, SQUARE)
    assert not rt._point_in_polygon(-10.0, 50.0, SQUARE)


def test_load_roi_none_when_absent():
    assert rt._load_roi({}) is None


def test_load_roi_rejects_short_polygon():
    with pytest.raises(ValueError):
        rt._load_roi({"roi": [[0, 0], [1, 1]]})


def test_velocity_gate_basic():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (2.0, 0.0, 12.0, 10.0)
    far = (1000.0, 0.0, 1010.0, 10.0)
    assert rt._velocity_gate_ok(a, b, dt_frames=1, max_px_per_frame=10.0)
    assert not rt._velocity_gate_ok(a, far, dt_frames=1, max_px_per_frame=10.0)


def test_velocity_gate_scales_with_dt():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (25.0, 0.0, 35.0, 10.0)
    assert not rt._velocity_gate_ok(a, b, dt_frames=1, max_px_per_frame=10.0)
    assert rt._velocity_gate_ok(a, b, dt_frames=5, max_px_per_frame=10.0)


def test_velocity_gate_disabled_when_zero():
    a = (0.0, 0.0, 10.0, 10.0)
    b = (10000.0, 0.0, 10010.0, 10.0)
    assert rt._velocity_gate_ok(a, b, dt_frames=1, max_px_per_frame=0.0)


def test_scale_gate_basic():
    a = (0.0, 0.0, 10.0, 10.0)
    similar = (0.0, 0.0, 12.0, 12.0)
    huge = (0.0, 0.0, 100.0, 100.0)
    assert rt._scale_gate_ok(a, similar, max_ratio=2.0)
    assert not rt._scale_gate_ok(a, huge, max_ratio=2.0)
    assert not rt._scale_gate_ok(huge, a, max_ratio=2.0)


def _row(frame_idx: int, track_id: int, cx: float, cy: float) -> dict:
    return {
        "video_id": "v",
        "frame_idx": frame_idx,
        "track_id": track_id,
        "class_name": "car",
        "confidence": 0.9,
        "bbox_xyxy": [cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0],
    }


def test_displacement_filter_keeps_moving():
    rows = [_row(i, 1, 100.0 + i * 10.0, 100.0) for i in range(20)]
    assert len(rt._apply_displacement_filter(rows, 20.0)) == 20


def test_displacement_filter_drops_stationary():
    rows = [_row(i, 1, 100.0, 100.0) for i in range(20)]
    assert rt._apply_displacement_filter(rows, 20.0) == []


def test_displacement_filter_drops_oscillating():
    # Odd count so first and last centers coincide: net is 0,
    # path is long. That is the phantom-track shape the filter
    # targets. With an even count the first and last centers
    # differ by one oscillation amplitude and the track would
    # (correctly) be kept as moving.
    rows = [_row(i, 1, 100.0 if i % 2 == 0 else 120.0, 100.0) for i in range(51)]
    assert rt._apply_displacement_filter(rows, 5.0) == []


def test_track_defaults_disable_new_gates():
    dets = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [45.0, 45.0, 55.0, 55.0],
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [395.0, 45.0, 405.0, 55.0],
        },
    ]
    tracks = rt.track(
        dets,
        track_thresh=0.0,
        iou_threshold=0.0,
        track_buffer=10,
        min_box_area=0.0,
        use_motion_prediction=False,
        max_centroid_distance_px=1000.0,
    )
    assert len({r["track_id"] for r in tracks}) == 1


def test_track_velocity_gate_prevents_far_match():
    dets = [
        {
            "video_id": "v",
            "frame_idx": 0,
            "track_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [45.0, 45.0, 55.0, 55.0],
        },
        {
            "video_id": "v",
            "frame_idx": 1,
            "track_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [395.0, 45.0, 405.0, 55.0],
        },
    ]
    tracks = rt.track(
        dets,
        track_thresh=0.0,
        iou_threshold=0.0,
        track_buffer=10,
        min_box_area=0.0,
        use_motion_prediction=False,
        max_centroid_distance_px=1000.0,
        max_velocity_px_per_frame=10.0,
    )
    assert len({r["track_id"] for r in tracks}) == 2


def test_track_displacement_filter_drops_stationary():
    dets = [
        {
            "video_id": "v",
            "frame_idx": i,
            "track_id": 0,
            "class_name": "car",
            "confidence": 0.9,
            "bbox_xyxy": [45.0, 45.0, 55.0, 55.0],
        }
        for i in range(20)
    ]
    tracks = rt.track(
        dets,
        track_thresh=0.0,
        iou_threshold=0.0,
        track_buffer=10,
        min_box_area=0.0,
        use_motion_prediction=False,
        max_centroid_distance_px=0.0,
        min_track_displacement_px=20.0,
    )
    assert tracks == []
