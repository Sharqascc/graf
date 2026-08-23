import numpy as np

from graf.calibration.world_coords import image_points_to_world, tracks_to_world_points
from graf.data.schema import TrackRecord


def test_image_points_to_world_identity():
    H = np.eye(3)
    pts = [(1.0, 2.0), (3.0, 4.0)]
    result = image_points_to_world(H, pts)
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)
    assert np.allclose(result, np.array(pts))


def test_image_points_to_world_scaling():
    H = np.array([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    result = image_points_to_world(H, [(2.0, 3.0)])
    assert np.allclose(result, [[4.0, 9.0]])


def test_tracks_to_world_points_footpoint_and_bbox():
    H = np.eye(3)
    t1 = TrackRecord(
        video_id="v1",
        frame_idx=1,
        track_id="a",
        class_name="car",
        bbox_xyxy=(0.0, 0.0, 4.0, 6.0),
        footpoint_img=(1.0, 2.0),
    )
    t2 = TrackRecord(
        video_id="v1",
        frame_idx=1,
        track_id="b",
        class_name="pedestrian",
        bbox_xyxy=(0.0, 10.0, 4.0, 16.0),
        footpoint_img=None,
    )
    out = tracks_to_world_points(H, [t1, t2])
    assert len(out) == 2
    assert out[0]["track_id"] == "a"
    assert out[0]["x_m"] == 1.0
    assert out[0]["y_m"] == 2.0
    assert out[0]["image_x"] == 1.0
    assert out[0]["image_y"] == 2.0
    # second uses bbox bottom center: x=(0+4)/2=2, y=16
    assert out[1]["x_m"] == 2.0
    assert out[1]["y_m"] == 16.0
    assert out[1]["image_x"] == 2.0
    assert out[1]["image_y"] == 16.0


def test_tracks_to_world_points_with_translation():
    H = np.array([[1.0, 0.0, 10.0], [0.0, 1.0, 20.0], [0.0, 0.0, 1.0]])
    t = TrackRecord(
        video_id="v2",
        frame_idx=2,
        track_id="c",
        class_name="bus",
        bbox_xyxy=(2.0, 3.0, 6.0, 7.0),
        footpoint_img=None,
    )
    out = tracks_to_world_points(H, [t])
    # bbox bottom center x=(2+6)/2=4, y=7, H maps (4,7)->(14,27)
    assert out[0]["x_m"] == 14.0
    assert out[0]["y_m"] == 27.0
