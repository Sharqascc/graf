import pytest

from graf.data.schema import (
    DetectionRecord,
    SSMEventRecord,
    TrackRecord,
    TrajectoryRecord,
    VideoRecord,
    track_to_detection,
)


# ---------------------------------------------------------------------------
# VideoRecord
# ---------------------------------------------------------------------------
def test_video_record_validate_ok():
    video = VideoRecord(
        video_id="v1",
        file_name="clip.mp4",
        location="site_a",
        fps=25.0,
        width=1920,
        height=1080,
    )
    video.validate()
    d = video.to_dict()
    assert d["video_id"] == "v1"
    assert d["fps"] == 25.0


def test_video_record_validate_bad_fps():
    video = VideoRecord("v1", "clip.mp4", "site_a", fps=0, width=1920, height=1080)
    with pytest.raises(ValueError, match="fps must be > 0"):
        video.validate()


def test_video_record_validate_bad_dimensions():
    video = VideoRecord("v1", "clip.mp4", "site_a", fps=25, width=0, height=1080)
    with pytest.raises(ValueError, match="width/height must be > 0"):
        video.validate()


# ---------------------------------------------------------------------------
# DetectionRecord
# ---------------------------------------------------------------------------
def test_detection_record_validate_bad_bbox_order():
    d = DetectionRecord(
        video_id="v1",
        frame_idx=0,
        actor_id=None,
        class_name="car",
        confidence=0.8,
        bbox_xyxy=(10.0, 10.0, 0.0, 20.0),
    )
    with pytest.raises(ValueError, match="Invalid bbox ordering"):
        d.validate()


def test_detection_record_validate_bad_confidence():
    d = DetectionRecord("v1", 0, None, "car", 1.5, (0, 0, 10, 10))
    with pytest.raises(ValueError, match="Confidence must be in"):
        d.validate()


def test_detection_record_validate_negative_frame():
    d = DetectionRecord("v1", -1, None, "car", 0.8, (0, 0, 10, 10))
    with pytest.raises(ValueError, match="frame_idx must be >= 0"):
        d.validate()


def test_detection_record_to_dict():
    d = DetectionRecord("v1", 2, "a", "car", 0.9, (0, 0, 10, 10))
    result = d.to_dict()
    assert result["actor_id"] == "a"
    assert result["bbox_xyxy"] == (0, 0, 10, 10)


# ---------------------------------------------------------------------------
# TrackRecord
# ---------------------------------------------------------------------------
def test_track_record_validate_bad_bbox_order():
    t = TrackRecord("v1", 1, "t1", "car", (10, 10, 0, 20))
    with pytest.raises(ValueError, match="Invalid bbox ordering"):
        t.validate()


def test_track_record_validate_bad_confidence():
    t = TrackRecord("v1", 1, "t1", "car", (0, 0, 10, 10), confidence=2.0)
    with pytest.raises(ValueError, match="Confidence must be in"):
        t.validate()


def test_track_record_validate_negative_frame():
    t = TrackRecord("v1", -1, "t1", "car", (0, 0, 10, 10))
    with pytest.raises(ValueError, match="frame_idx must be >= 0"):
        t.validate()


def test_track_record_to_dict():
    t = TrackRecord(
        "v1",
        3,
        "t1",
        "pedestrian",
        (1, 2, 11, 22),
        footpoint_img=(6.0, 22.0),
        confidence=0.7,
        occluded=True,
    )
    d = t.to_dict()
    assert d["track_id"] == "t1"
    assert d["footpoint_img"] == (6.0, 22.0)
    assert d["occluded"] is True


# ---------------------------------------------------------------------------
# TrajectoryRecord
# ---------------------------------------------------------------------------
def test_trajectory_record_validate_ok():
    traj = TrajectoryRecord(
        video_id="v1",
        frame_idx=1,
        track_id="t1",
        class_name="car",
        x_m=10.0,
        y_m=20.0,
        t_sec=0.04,
        speed_mps=5.0,
    )
    traj.validate()
    d = traj.to_dict()
    assert d["x_m"] == 10.0
    assert d["speed_mps"] == 5.0


def test_trajectory_record_validate_bad_frame():
    traj = TrajectoryRecord("v1", -1, "t1", "car", 0, 0, 0)
    with pytest.raises(ValueError, match="frame_idx must be >= 0"):
        traj.validate()


def test_trajectory_record_validate_negative_t():
    traj = TrajectoryRecord("v1", 0, "t1", "car", 0, 0, -0.1)
    with pytest.raises(ValueError, match="t_sec must be >= 0"):
        traj.validate()


# ---------------------------------------------------------------------------
# SSMEventRecord
# ---------------------------------------------------------------------------
def test_ssm_event_record_validate_ok():
    event = SSMEventRecord(
        video_id="v1",
        event_id="e1",
        metric_name="TTC",
        track_id_a="a",
        track_id_b="b",
        start_frame=1,
        end_frame=5,
        min_value=0.5,
        threshold=1.0,
        severity="high",
    )
    event.validate()
    d = event.to_dict()
    assert d["metric_name"] == "TTC"
    assert d["severity"] == "high"


def test_ssm_event_record_validate_bad_frame_order():
    event = SSMEventRecord(
        video_id="v1",
        event_id="e1",
        metric_name="TTC",
        track_id_a="a",
        track_id_b="b",
        start_frame=10,
        end_frame=2,
        min_value=0.5,
    )
    with pytest.raises(ValueError, match="start_frame must be <= end_frame"):
        event.validate()


# ---------------------------------------------------------------------------
# track_to_detection confidence override
# ---------------------------------------------------------------------------
def test_track_to_detection_respects_existing_confidence():
    track = TrackRecord(
        "v1",
        1,
        "t1",
        "car",
        (0, 0, 10, 10),
        confidence=0.6,
    )
    det = track_to_detection(track, confidence=0.9)
    assert det.confidence == 0.6  # existing should win


def test_track_to_detection_uses_provided_confidence_if_none():
    track = TrackRecord("v1", 1, "t1", "car", (0, 0, 10, 10), confidence=None)
    det = track_to_detection(track, confidence=0.75)
    assert det.confidence == 0.75
