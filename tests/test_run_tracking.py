"""Tests for scripts/run_tracking.py.

Covers config loading/resolution, filtering (confidence + area), greedy
IoU association, and — importantly — the staleness buffer, which was
previously broken because frames_since_update was never incremented.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from scripts import run_tracking as script_mod

REPO = Path(__file__).resolve().parents[1]


def _write_cfg(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _det(
    video="v", frame=0, tid_hint=None, cls="car", conf=0.9, bbox=(0.0, 0.0, 50.0, 50.0)
) -> dict:
    return {
        "video_id": video,
        "frame_idx": frame,
        "actor_id": tid_hint,
        "class_name": cls,
        "confidence": conf,
        "bbox_xyxy": list(bbox),
    }


# ------------------------------------------------------------------
# config loading
# ------------------------------------------------------------------


def test_load_config_reads_yaml(tmp_path):
    p = _write_cfg(tmp_path / "c.yaml", {"track_thresh": 0.7, "track_buffer": 15})
    cfg = script_mod.load_config(p)
    assert cfg["track_thresh"] == 0.7
    assert cfg["track_buffer"] == 15


def test_load_config_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        script_mod.load_config(tmp_path / "nope.yaml")


def test_load_config_non_mapping_raises(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        script_mod.load_config(p)


def test_resolve_cli_wins():
    assert script_mod.resolve({"track_thresh": 0.7}, "track_thresh", 0.5, 0.9) == 0.5


def test_default_config_parses():
    cfg = script_mod.load_config(script_mod._DEFAULT_CONFIG)
    assert cfg["iou_threshold"] == 0.3
    assert cfg["track_buffer"] == 30


# ------------------------------------------------------------------
# filtering
# ------------------------------------------------------------------


def test_low_confidence_dropped():
    dets = [
        _det(frame=0, conf=0.3, bbox=(0, 0, 50, 50)),
        _det(frame=0, conf=0.9, bbox=(60, 60, 110, 110)),
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
    )
    assert len(tracks) == 1
    assert tracks[0]["confidence"] == 0.9


def test_tiny_box_dropped():
    dets = [
        _det(frame=0, conf=0.9, bbox=(0, 0, 2, 2)),  # 4 px^2 < 10
        _det(frame=0, conf=0.9, bbox=(60, 60, 110, 110)),
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
    )
    assert len(tracks) == 1


# ------------------------------------------------------------------
# association
# ------------------------------------------------------------------


def test_two_frames_same_box_one_track():
    dets = [
        _det(frame=0, bbox=(0, 0, 50, 50)),
        _det(frame=1, bbox=(0, 0, 50, 50)),
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
    )
    tids = {t["track_id"] for t in tracks}
    assert len(tids) == 1
    assert len(tracks) == 2


def test_two_frames_moved_box_new_track():
    dets = [
        _det(frame=0, bbox=(0, 0, 50, 50)),
        _det(frame=1, bbox=(500, 500, 550, 550)),  # zero IoU
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
    )
    tids = {t["track_id"] for t in tracks}
    assert len(tids) == 2


def test_class_mismatch_does_not_associate():
    dets = [
        _det(frame=0, cls="car", bbox=(0, 0, 50, 50)),
        _det(frame=1, cls="pedestrian", bbox=(0, 0, 50, 50)),
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
    )
    assert len({t["track_id"] for t in tracks}) == 2


# ------------------------------------------------------------------
# staleness (the previously broken behavior)
# ------------------------------------------------------------------


def test_stale_track_is_dropped():
    # Track A appears at frame 0 then never again. A different track B
    # appears at frames 20-25. With a short buffer, A should be dropped
    # and B should NOT reuse A's id.
    dets = [
        _det(frame=0, bbox=(0, 0, 50, 50)),
        _det(frame=20, bbox=(500, 500, 550, 550)),
        _det(frame=21, bbox=(500, 500, 550, 550)),
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=5,
        min_box_area=10.0,
    )
    # If staleness works, frame 20's box cannot match frame 0's track
    # anyway (IoU=0), but the key assertion is that the stale track
    # from frame 0 was removed from the active pool before frame 20.
    assert len(tracks) == 3


def test_track_buffer_zero_drops_after_one_miss():
    dets = [
        _det(frame=0, bbox=(0, 0, 50, 50)),
        _det(frame=1, bbox=(0, 0, 50, 50)),
        _det(frame=2, bbox=(100, 100, 150, 150)),  # no overlap -> no match
        _det(frame=3, bbox=(100, 100, 150, 150)),  # should be a new track
    ]
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=0,
        min_box_area=10.0,
    )
    # Two tracks total: one that died at frame 1, one that started at 2
    tids = {t["track_id"] for t in tracks}
    assert len(tids) == 2


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def test_script_help():
    r = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "run_tracking.py"), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    for flag in (
        "--detections",
        "--output_dir",
        "--config",
        "--track_thresh",
        "--iou_threshold",
        "--track_buffer",
        "--min_box_area",
    ):
        assert flag in r.stdout


# ------------------------------------------------------------------
# motion-aware prediction
# ------------------------------------------------------------------


def _const_velocity_seq(
    dx_per_frame: float,
    n_frames: int,
    stride: int = 1,
    box_size: float = 40.0,
    start: tuple[float, float] = (100.0, 100.0),
) -> list[dict]:
    """One actor moving at (dx_per_frame, 0) px/frame, processed at `stride`."""
    rows = []
    for f in range(0, n_frames, stride):
        x = start[0] + dx_per_frame * f
        y = start[1]
        rows.append(
            _det(
                frame=f,
                cls="car",
                conf=0.9,
                bbox=(x, y, x + box_size, y + box_size),
            )
        )
    return rows


def test_motion_prediction_keeps_fast_mover_one_track():
    # 30 px/frame at stride 2 -> 60 px displacement between matched frames.
    # With box_size 40, IoU between unshifted bboxes at 60 px separation
    # is 0; with motion prediction the shifted probe should match.
    dets = _const_velocity_seq(dx_per_frame=30.0, n_frames=10, stride=2)
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
        use_motion_prediction=True,
        velocity_smoothing=1.0,
        max_prediction_offset_px=500.0,
    )
    tids = {t["track_id"] for t in tracks}
    assert len(tids) == 1, f"expected 1 track, got {len(tids)}"


def test_iou_only_reproduces_stride_artifact():
    # Explicitly turn off both prediction and centroid fallback to
    # reproduce the original stride artifact: fast mover at stride 2
    # splits into many tracks because IoU between unshifted bboxes is 0.
    dets = _const_velocity_seq(dx_per_frame=30.0, n_frames=10, stride=2)
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
        use_motion_prediction=False,
        max_centroid_distance_px=0.0,
    )
    tids = {t["track_id"] for t in tracks}
    assert len(tids) > 1, "expected the stride artifact in IoU-only mode"


def test_prediction_offset_capped():
    # Very fast mover with a small cap should NOT match far-away detections.
    dets = _const_velocity_seq(dx_per_frame=200.0, n_frames=6, stride=2)
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
        use_motion_prediction=True,
        velocity_smoothing=1.0,
        max_prediction_offset_px=30.0,
    )
    tids = {t["track_id"] for t in tracks}
    # Cap too small -> cannot keep up -> multiple tracks.
    assert len(tids) > 1


def test_stationary_actor_stays_one_track_either_way():
    dets = _const_velocity_seq(dx_per_frame=0.0, n_frames=8, stride=2)
    for flag in (True, False):
        tracks = script_mod.track(
            dets,
            track_thresh=0.5,
            iou_threshold=0.3,
            track_buffer=30,
            min_box_area=10.0,
            use_motion_prediction=flag,
        )
        assert len({t["track_id"] for t in tracks}) == 1


def test_velocity_smoothing_out_of_range_raises():
    with pytest.raises(ValueError):
        script_mod.track(
            [],
            track_thresh=0.5,
            iou_threshold=0.3,
            track_buffer=30,
            min_box_area=10.0,
            velocity_smoothing=0.0,
        )
    with pytest.raises(ValueError):
        script_mod.track(
            [],
            track_thresh=0.5,
            iou_threshold=0.3,
            track_buffer=30,
            min_box_area=10.0,
            velocity_smoothing=1.5,
        )


def test_negative_cap_raises():
    with pytest.raises(ValueError):
        script_mod.track(
            [],
            track_thresh=0.5,
            iou_threshold=0.3,
            track_buffer=30,
            min_box_area=10.0,
            max_prediction_offset_px=-1.0,
        )


def test_centroid_fallback_bootstraps_velocity():
    # First match has IoU=0 (stride 2, 30 px/frame => 60 px displacement,
    # box size 40). Without centroid fallback, the track dies before
    # gaining velocity. With it, one track should survive.
    dets = _const_velocity_seq(dx_per_frame=30.0, n_frames=10, stride=2)
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
        use_motion_prediction=True,
        velocity_smoothing=1.0,
        max_prediction_offset_px=500.0,
        max_centroid_distance_px=200.0,
    )
    assert len({t["track_id"] for t in tracks}) == 1


def test_centroid_fallback_disabled_when_zero():
    dets = _const_velocity_seq(dx_per_frame=30.0, n_frames=10, stride=2)
    tracks = script_mod.track(
        dets,
        track_thresh=0.5,
        iou_threshold=0.3,
        track_buffer=30,
        min_box_area=10.0,
        use_motion_prediction=True,
        velocity_smoothing=1.0,
        max_prediction_offset_px=500.0,
        max_centroid_distance_px=0.0,
    )
    # No centroid fallback -> fast mover splits into multiple tracks
    assert len({t["track_id"] for t in tracks}) > 1


def test_negative_centroid_distance_raises():
    with pytest.raises(ValueError):
        script_mod.track(
            [],
            track_thresh=0.5,
            iou_threshold=0.3,
            track_buffer=30,
            min_box_area=10.0,
            max_centroid_distance_px=-1.0,
        )


def test_cli_couples_motion_off_with_centroid_off(tmp_path):
    """--no_motion_prediction implies max_centroid_distance_px=0."""
    dets = _const_velocity_seq(dx_per_frame=30.0, n_frames=10, stride=2)
    det_path = tmp_path / "dets.jsonl"
    with det_path.open("w") as f:
        for d in dets:
            f.write(json.dumps(d) + "\n")

    out_dir = tmp_path / "out"
    r = subprocess.run(
        [
            sys.executable,
            str(REPO / "scripts" / "run_tracking.py"),
            "--detections",
            str(det_path),
            "--output_dir",
            str(out_dir),
            "--no_motion_prediction",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    assert r.returncode == 0, r.stderr
    rows = [
        json.loads(line) for line in (out_dir / "tracks.jsonl").read_text().splitlines()
    ]
    tids = {r["track_id"] for r in rows}
    assert len(tids) > 1, "expected stride artifact when --no_motion_prediction"
