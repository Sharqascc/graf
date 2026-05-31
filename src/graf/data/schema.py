from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional, Tuple


BBox = Tuple[float, float, float, float]
Point2D = Tuple[float, float]


@dataclass(slots=True)
class VideoRecord:
    video_id: str
    file_name: str
    location: str
    fps: float
    width: int
    height: int
    start_time: Optional[str] = None
    duration_sec: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be > 0, got {self.fps}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"width/height must be > 0, got {(self.width, self.height)}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DetectionRecord:
    video_id: str
    frame_idx: int
    actor_id: Optional[str]
    class_name: str
    confidence: float
    bbox_xyxy: BBox

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox ordering: {self.bbox_xyxy}")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
        if self.frame_idx < 0:
            raise ValueError(f"frame_idx must be >= 0, got {self.frame_idx}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrackRecord:
    video_id: str
    frame_idx: int
    track_id: str
    class_name: str
    bbox_xyxy: BBox
    footpoint_img: Optional[Point2D] = None
    confidence: Optional[float] = None
    occluded: bool = False

    def validate(self) -> None:
        x1, y1, x2, y2 = self.bbox_xyxy
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid bbox ordering: {self.bbox_xyxy}")
        if self.frame_idx < 0:
            raise ValueError(f"frame_idx must be >= 0, got {self.frame_idx}")
        if self.confidence is not None and not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrajectoryRecord:
    video_id: str
    frame_idx: int
    track_id: str
    class_name: str
    x_m: float
    y_m: float
    t_sec: float
    speed_mps: Optional[float] = None
    heading_rad: Optional[float] = None
    accel_mps2: Optional[float] = None
    source: str = "homography"

    def validate(self) -> None:
        if self.frame_idx < 0:
            raise ValueError(f"frame_idx must be >= 0, got {self.frame_idx}")
        if self.t_sec < 0:
            raise ValueError(f"t_sec must be >= 0, got {self.t_sec}")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SSMEventRecord:
    video_id: str
    event_id: str
    metric_name: str
    track_id_a: str
    track_id_b: str
    start_frame: int
    end_frame: int
    min_value: float
    threshold: Optional[float] = None
    severity: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.start_frame > self.end_frame:
            raise ValueError(
                f"start_frame must be <= end_frame, got {(self.start_frame, self.end_frame)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def bbox_bottom_center(bbox_xyxy: BBox) -> Point2D:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)


def track_to_detection(track: TrackRecord, confidence: float = 1.0) -> DetectionRecord:
    return DetectionRecord(
        video_id=track.video_id,
        frame_idx=track.frame_idx,
        actor_id=track.track_id,
        class_name=track.class_name,
        confidence=confidence if track.confidence is None else track.confidence,
        bbox_xyxy=track.bbox_xyxy,
    )
