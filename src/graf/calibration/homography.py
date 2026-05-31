from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import cv2
import numpy as np


Point2D = Tuple[float, float]


@dataclass(slots=True)
class HomographyResult:
    H: np.ndarray
    image_points: np.ndarray
    world_points: np.ndarray
    rms_error: float


def _to_array(points: Sequence[Point2D]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Points must have shape (N, 2)")
    return arr


def project_points(H: np.ndarray, image_points: Sequence[Point2D]) -> np.ndarray:
    pts = _to_array(image_points)
    pts_h = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    warped = (H @ pts_h.T).T
    denom = warped[:, 2:3]
    if np.any(np.isclose(denom, 0.0)):
        raise ZeroDivisionError("Homogeneous coordinate contains zero scale term")
    return warped[:, :2] / denom


def fit_homography(
    image_points: Sequence[Point2D],
    world_points: Sequence[Point2D],
) -> HomographyResult:
    img = _to_array(image_points)
    world = _to_array(world_points)

    if len(img) < 4 or len(world) < 4:
        raise ValueError("At least 4 point correspondences are required")
    if len(img) != len(world):
        raise ValueError("image_points and world_points must have same length")

    H, _ = cv2.findHomography(img, world, method=0)
    if H is None:
        raise RuntimeError("Failed to estimate homography matrix")

    projected = project_points(H, img)
    err = np.sqrt(np.mean(np.sum((projected - world) ** 2, axis=1)))

    return HomographyResult(
        H=H,
        image_points=img,
        world_points=world,
        rms_error=float(err),
    )


def invert_homography(H: np.ndarray) -> np.ndarray:
    return np.linalg.inv(H)


def world_to_image(H: np.ndarray, world_points: Sequence[Point2D]) -> np.ndarray:
    H_inv = invert_homography(H)
    return project_points(H_inv, world_points)
