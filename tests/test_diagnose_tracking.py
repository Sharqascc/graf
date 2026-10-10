"""Tests for scripts/diagnose_tracking.py (Phase 0 of the SCI plan).

The script is loaded by path via importlib, matching the pattern used
for scripts/track_quality_summary.py. Tests cover the pure functions:
_center, _area, classify_motion, diagnose_track, diagnose_all, and
summarise. They do not test the CLI wrapper or file IO beyond the
minimal case.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "diagnose_tracking_under_test",
    _REPO / "scripts" / "diagnose_tracking.py",
)
assert _SPEC is not None
assert _SPEC.loader is not None
dt = importlib.util.module_from_spec(_SPEC)
# Register in sys.modules before exec so dataclass(slots=True) can
# resolve the class's own module by name during decoration.
sys.modules[_SPEC.name] = dt
_SPEC.loader.exec_module(dt)


def _row(
    frame_idx: int,
    track_id: int = 1,
    cx: float = 100.0,
    cy: float = 100.0,
    w: float = 20.0,
    h: float = 20.0,
    class_name: str = "car",
) -> dict:
    return {
        "frame_idx": frame_idx,
        "track_id": track_id,
        "class_name": class_name,
        "confidence": 0.9,
        "bbox_xyxy": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
    }


def test_center_and_area():
    bbox = (10.0, 20.0, 30.0, 60.0)
    cx, cy = dt._center(bbox)
    assert cx == 20.0
    assert cy == 40.0
    assert dt._area(bbox) == 20.0 * 40.0


def test_classify_moving():
    assert dt.classify_motion(50.0, 1.2) == "moving"


def test_classify_stationary_held():
    assert dt.classify_motion(1.0, 1.5) == "stationary_held"


def test_classify_stationary_jumping():
    assert dt.classify_motion(1.0, 10.0) == "stationary_jumping"


def test_diagnose_held_track():
    rows = [_row(i, cx=100.0, cy=100.0) for i in range(50)]
    d = dt.diagnose_track(1, rows)
    assert d.n_frames == 50
    assert d.classification == "stationary_held"
    assert d.net_displacement_px < 1.0


def test_diagnose_moving_track():
    rows = [_row(i, cx=100.0 + i * 5.0, cy=100.0) for i in range(50)]
    d = dt.diagnose_track(1, rows)
    assert d.classification == "moving"
    assert d.net_displacement_px > 200.0


def test_diagnose_jumping_track():
    # Alternates between two far-apart positions; net stays small,
    # path is large.
    rows = []
    for i in range(50):
        cx = 100.0 if i % 2 == 0 else 200.0
        rows.append(_row(i, cx=cx, cy=100.0))
    d = dt.diagnose_track(1, rows)
    assert d.classification == "stationary_jumping"
    assert d.jitter_ratio > 3.0


def test_diagnose_all_sorts_by_length():
    rows = (
        [_row(i, track_id=1) for i in range(10)]
        + [_row(i, track_id=2) for i in range(30)]
        + [_row(i, track_id=3) for i in range(20)]
    )
    diags = dt.diagnose_all(rows)
    assert [d.track_id for d in diags] == [2, 3, 1]


def test_diagnose_all_empty():
    assert dt.diagnose_all([]) == []


def test_summarise_groups_by_class():
    rows = (
        [_row(i, track_id=1, class_name="car") for i in range(10)]
        + [_row(i, track_id=2, class_name="car") for i in range(20)]
        + [_row(i, track_id=3, class_name="two_wheeler") for i in range(15)]
    )
    diags = dt.diagnose_all(rows)
    s = dt.summarise(diags)
    assert s["car"]["n_tracks"] == 2
    assert s["two_wheeler"]["n_tracks"] == 1


def test_interpret_returns_string():
    rows = [_row(i, cx=100.0, cy=100.0) for i in range(50)]
    diags = dt.diagnose_all(rows)
    out = dt.interpret(diags)
    assert isinstance(out, str)
    assert "stationary_held" in out


def test_interpret_empty():
    out = dt.interpret([])
    assert "No tracks" in out
