"""Property-based tests for ``scripts/run_tracking.py``.

Covers the IoU geometry primitives, the association score, and the
greedy motion-aware tracker's structural invariants. The module under
test is a *script*, not an importable package module, so it is loaded
from its file path via importlib.

Complements ``tests/test_property_based.py``, which covers homography,
kinematics, TTC, DRAC, PET, and event mining, but has no coverage of
tracking at all.
"""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ── Load scripts/run_tracking.py as a module ─────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "run_tracking_under_test", _ROOT / "scripts" / "run_tracking.py"
)
assert _SPEC is not None and _SPEC.loader is not None
run_tracking = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_tracking)


# ── Strategies ───────────────────────────────────────────────────────
_coord = st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False)
_side = st.floats(0.5, 500.0, allow_nan=False, allow_infinity=False)
_shift = st.floats(-50.0, 50.0, allow_nan=False, allow_infinity=False)
_conf = st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False)
_classes = st.sampled_from(["car", "pedestrian", "two_wheeler", "bus", "bicycle"])


@st.composite
def boxes(draw):
    x1 = draw(_coord)
    y1 = draw(_coord)
    w = draw(_side)
    h = draw(_side)
    return (x1, y1, x1 + w, y1 + h)


@st.composite
def tracker_boxes(draw):
    """Tight boxes so overlaps actually occur in tracker tests."""
    x1 = draw(st.floats(-5.0, 5.0, allow_nan=False))
    y1 = draw(st.floats(-5.0, 5.0, allow_nan=False))
    w = draw(st.floats(2.0, 8.0, allow_nan=False))
    h = draw(st.floats(2.0, 8.0, allow_nan=False))
    return (x1, y1, x1 + w, y1 + h)


@st.composite
def detection_rows(draw):
    return {
        "video_id": "v",
        "frame_idx": draw(st.integers(0, 10)),
        "class_name": draw(_classes),
        "confidence": draw(_conf),
        "bbox_xyxy": draw(tracker_boxes()),
    }


@st.composite
def detection_lists(draw, max_n=15):
    return draw(st.lists(detection_rows(), min_size=0, max_size=max_n))


def _run(dets, **overrides):
    kwargs = {
        "track_thresh": 0.0,
        "iou_threshold": 0.3,
        "track_buffer": 10,
        "min_box_area": 0.0,
        "use_motion_prediction": True,
        "velocity_smoothing": 0.5,
        "max_prediction_offset_px": 300.0,
        "max_centroid_distance_px": 200.0,
    }
    kwargs.update(overrides)
    return run_tracking.track(dets, **kwargs)


# ── IoU ──────────────────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(a=boxes(), b=boxes())
@settings(max_examples=100, deadline=None)
def test_iou_is_in_unit_interval(a, b):
    assert 0.0 <= run_tracking.iou(a, b) <= 1.0


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes())
@settings(max_examples=100, deadline=None)
def test_iou_is_symmetric(a, b):
    assert run_tracking.iou(a, b) == pytest.approx(
        run_tracking.iou(b, a), rel=1e-9, abs=1e-12
    )


@pytest.mark.hypothesis
@given(b=boxes())
@settings(max_examples=50, deadline=None)
def test_iou_with_self_is_one(b):
    # Strategy guarantees strictly positive width and height.
    assert run_tracking.iou(b, b) == pytest.approx(1.0, rel=1e-12, abs=1e-12)


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes(), dx=_shift, dy=_shift)
@settings(max_examples=100, deadline=None)
def test_iou_joint_translation_invariant(a, b, dx, dy):
    a2 = run_tracking.shift_bbox(a, dx, dy)
    b2 = run_tracking.shift_bbox(b, dx, dy)
    assert run_tracking.iou(a, b) == pytest.approx(
        run_tracking.iou(a2, b2), rel=1e-9, abs=1e-9
    )


# ── Box primitives ───────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(b=boxes())
@settings(max_examples=50, deadline=None)
def test_box_area_is_non_negative(b):
    assert run_tracking.box_area(b) >= 0.0


@pytest.mark.hypothesis
@given(b=boxes(), dx=_shift, dy=_shift)
@settings(max_examples=50, deadline=None)
def test_box_area_translation_invariant(b, dx, dy):
    assert run_tracking.box_area(run_tracking.shift_bbox(b, dx, dy)) == pytest.approx(
        run_tracking.box_area(b), rel=1e-9, abs=1e-9
    )


@pytest.mark.hypothesis
@given(b=boxes(), dx=_shift, dy=_shift)
@settings(max_examples=50, deadline=None)
def test_shift_bbox_translates_center_by_delta(b, dx, dy):
    cx, cy = run_tracking.bbox_center(b)
    sx, sy = run_tracking.bbox_center(run_tracking.shift_bbox(b, dx, dy))
    assert sx == pytest.approx(cx + dx, rel=1e-9, abs=1e-9)
    assert sy == pytest.approx(cy + dy, rel=1e-9, abs=1e-9)


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes())
@settings(max_examples=100, deadline=None)
def test_centroid_distance_symmetric(a, b):
    assert run_tracking.centroid_distance(a, b) == pytest.approx(
        run_tracking.centroid_distance(b, a), rel=1e-9, abs=1e-9
    )


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes(), dx=_shift, dy=_shift)
@settings(max_examples=50, deadline=None)
def test_centroid_distance_translation_invariant(a, b, dx, dy):
    a2 = run_tracking.shift_bbox(a, dx, dy)
    b2 = run_tracking.shift_bbox(b, dx, dy)
    assert run_tracking.centroid_distance(a, b) == pytest.approx(
        run_tracking.centroid_distance(a2, b2), rel=1e-9, abs=1e-9
    )


# ── association_score ────────────────────────────────────────────────
@pytest.mark.hypothesis
@given(a=boxes(), b=boxes(), max_cdist=st.floats(1.0, 1e3))
@settings(max_examples=100, deadline=None)
def test_association_score_at_least_iou(a, b, max_cdist):
    score, iou_v = run_tracking.association_score(a, b, max_cdist)
    assert score >= iou_v - 1e-12


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes(), max_cdist=st.floats(1.0, 1e3))
@settings(max_examples=100, deadline=None)
def test_association_score_in_unit_interval(a, b, max_cdist):
    score, iou_v = run_tracking.association_score(a, b, max_cdist)
    assert 0.0 <= score <= 1.0
    assert 0.0 <= iou_v <= 1.0


@pytest.mark.hypothesis
@given(a=boxes(), b=boxes(), max_cdist=st.floats(-100.0, 0.0))
@settings(max_examples=50, deadline=None)
def test_association_score_disabled_centroid_equals_iou(a, b, max_cdist):
    score, iou_v = run_tracking.association_score(a, b, max_cdist)
    assert math.isclose(score, iou_v, rel_tol=1e-9, abs_tol=1e-12)


# ── Tracker: structural invariants ───────────────────────────────────
@pytest.mark.hypothesis
@given(dets=detection_lists())
@settings(max_examples=50, deadline=None)
def test_tracker_ids_unique_per_frame(dets):
    per_frame: dict[int, list[int]] = {}
    for r in _run(dets):
        per_frame.setdefault(r["frame_idx"], []).append(r["track_id"])
    for fid, tids in per_frame.items():
        assert len(tids) == len(set(tids)), f"duplicate track ID in frame {fid}"


@pytest.mark.hypothesis
@given(dets=detection_lists())
@settings(max_examples=50, deadline=None)
def test_tracker_deterministic(dets):
    assert _run(dets) == _run(dets)


@pytest.mark.hypothesis
@given(dets=detection_lists())
@settings(max_examples=50, deadline=None)
def test_tracker_output_rows_are_real_detections(dets):
    """Every output row must be traceable back to an input detection row."""
    input_keys = {
        (d["frame_idx"], d["class_name"], d["confidence"], d["bbox_xyxy"]) for d in dets
    }
    for r in _run(dets):
        key = (r["frame_idx"], r["class_name"], r["confidence"], r["bbox_xyxy"])
        assert key in input_keys


@pytest.mark.hypothesis
@given(dets=detection_lists(), thr=st.floats(0.5, 1.0))
@settings(max_examples=50, deadline=None)
def test_tracker_respects_track_thresh(dets, thr):
    for r in _run(dets, track_thresh=thr):
        assert r["confidence"] >= thr


@pytest.mark.hypothesis
@given(dets=detection_lists(), min_area=st.floats(10.0, 500.0))
@settings(max_examples=50, deadline=None)
def test_tracker_respects_min_box_area(dets, min_area):
    for r in _run(dets, min_box_area=min_area):
        assert run_tracking.box_area(r["bbox_xyxy"]) >= min_area


@pytest.mark.hypothesis
@given(dets=detection_lists())
@settings(max_examples=50, deadline=None)
def test_tracker_never_merges_across_classes(dets):
    """A single track_id must map to exactly one class_name."""
    id_to_class: dict[int, str] = {}
    for r in _run(dets):
        tid, cls = r["track_id"], r["class_name"]
        if tid in id_to_class:
            assert id_to_class[tid] == cls, f"track {tid} changed class"
        else:
            id_to_class[tid] = cls


@pytest.mark.hypothesis
@given(dets=detection_lists())
@settings(max_examples=50, deadline=None)
def test_tracker_single_row_per_track_per_frame(dets):
    seen: set[tuple[int, int]] = set()
    for r in _run(dets):
        key = (r["track_id"], r["frame_idx"])
        assert key not in seen, (
            f"track {r['track_id']} emitted twice in frame {r['frame_idx']}"
        )
        seen.add(key)


# ── Tracker: argument validation ─────────────────────────────────────
@pytest.mark.hypothesis
@given(dets=detection_lists(max_n=3), alpha=st.floats(-5.0, 0.0))
@settings(max_examples=20, deadline=None)
def test_tracker_rejects_bad_velocity_smoothing(dets, alpha):
    with pytest.raises(ValueError):
        _run(dets, velocity_smoothing=alpha)


@pytest.mark.hypothesis
@given(dets=detection_lists(max_n=3), offset=st.floats(-100.0, -1e-6))
@settings(max_examples=20, deadline=None)
def test_tracker_rejects_negative_max_prediction_offset(dets, offset):
    with pytest.raises(ValueError):
        _run(dets, max_prediction_offset_px=offset)


@pytest.mark.hypothesis
@given(dets=detection_lists(max_n=3), offset=st.floats(-100.0, -1e-6))
@settings(max_examples=20, deadline=None)
def test_tracker_rejects_negative_max_centroid_distance(dets, offset):
    with pytest.raises(ValueError):
        _run(dets, max_centroid_distance_px=offset)
