from graf.data.schema import DetectionRecord, TrackRecord, bbox_bottom_center, track_to_detection


def test_bbox_bottom_center():
    bbox = (10.0, 20.0, 30.0, 60.0)
    x, y = bbox_bottom_center(bbox)
    assert x == 20.0
    assert y == 60.0


def test_detection_validation():
    d = DetectionRecord(
        video_id="vid1",
        frame_idx=1,
        actor_id=None,
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(0.0, 0.0, 10.0, 10.0),
    )
    d.validate()


def test_track_to_detection():
    track = TrackRecord(
        video_id="vid1",
        frame_idx=5,
        track_id="t1",
        class_name="car",
        bbox_xyxy=(1.0, 2.0, 11.0, 22.0),
    )
    det = track_to_detection(track)
    assert det.actor_id == "t1"
    assert det.class_name == "car"
    assert det.bbox_xyxy == (1.0, 2.0, 11.0, 22.0)
