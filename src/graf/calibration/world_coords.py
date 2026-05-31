from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

from graf.calibration.homography import project_points
from graf.data.schema import TrackRecord, bbox_bottom_center


Point2D = Tuple[float, float]


def image_points_to_world(H: np.ndarray, points: Sequence[Point2D]) -> np.ndarray:
    return project_points(H, points)


def tracks_to_world_points(H: np.ndarray, tracks: Sequence[TrackRecord]) -> List[dict]:
    image_pts = [
        track.footpoint_img if track.footpoint_img is not None else bbox_bottom_center(track.bbox_xyxy)
        for track in tracks
    ]
    world_pts = image_points_to_world(H, image_pts)

    output = []
    for track, (x_m, y_m), img_pt in zip(tracks, world_pts, image_pts):
        output.append({
            "video_id": track.video_id,
            "frame_idx": track.frame_idx,
            "track_id": track.track_id,
            "class_name": track.class_name,
            "image_x": float(img_pt[0]),
            "image_y": float(img_pt[1]),
            "x_m": float(x_m),
            "y_m": float(y_m),
        })
    return output
