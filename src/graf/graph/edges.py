from __future__ import annotations

import math
from typing import Any

ACTOR_CLASSES = [
    "car",
    "pedestrian",
    "two_wheeler",
    "auto_rickshaw",
    "bus",
    "truck",
    "bicycle",
    "other",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(ACTOR_CLASSES)}

SIZE_PRIORS = {
    "pedestrian": 0.10,
    "bicycle": 0.20,
    "two_wheeler": 0.25,
    "auto_rickshaw": 0.40,
    "car": 0.60,
    "truck": 0.90,
    "bus": 1.00,
    "other": 0.30,
}

EDGE_FEATURE_ORDER = [
    "dx",
    "dy",
    "distance",
    "dvx",
    "dvy",
    "relative_speed",
    "closing_speed",
    "bearing_sin",
    "bearing_cos",
    "rel_heading_sin",
    "rel_heading_cos",
    "ttc",
    "same_class",
    "size_src",
    "size_dst",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def wrap_angle_rad(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def infer_actor_class(actor: dict[str, Any]) -> str:
    raw = (
        actor.get("actor_class")
        or actor.get("class_name")
        or actor.get("class")
        or actor.get("label")
        or "other"
    )
    raw = str(raw).strip().lower()
    return raw if raw in CLASS_TO_INDEX else "other"


def compute_closing_speed(
    dx: float,
    dy: float,
    dvx: float,
    dvy: float,
    distance: float,
) -> float:
    if distance <= 1e-8:
        return 0.0
    ux = dx / distance
    uy = dy / distance
    return -(dvx * ux + dvy * uy)


def compute_ttc_simple(
    x1: float,
    y1: float,
    vx1: float,
    vy1: float,
    x2: float,
    y2: float,
    vx2: float,
    vy2: float,
    collision_radius: float = 1.5,
    max_ttc: float = 10.0,
) -> float:
    rx = x2 - x1
    ry = y2 - y1
    rvx = vx2 - vx1
    rvy = vy2 - vy1

    a = rvx * rvx + rvy * rvy
    if a < 1e-8:
        return max_ttc

    b = 2.0 * (rx * rvx + ry * rvy)
    c = rx * rx + ry * ry - collision_radius * collision_radius

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return max_ttc

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if 0.0 <= t <= max_ttc]
    if not candidates:
        return max_ttc
    return min(candidates)


def build_edge_feature(
    src: dict[str, Any],
    dst: dict[str, Any],
    *,
    collision_radius: float = 1.5,
    max_ttc: float = 10.0,
) -> dict[str, float]:
    x1 = safe_float(src.get("x_m", src.get("x", 0.0)))
    y1 = safe_float(src.get("y_m", src.get("y", 0.0)))
    x2 = safe_float(dst.get("x_m", dst.get("x", 0.0)))
    y2 = safe_float(dst.get("y_m", dst.get("y", 0.0)))

    vx1 = safe_float(src.get("vx", 0.0))
    vy1 = safe_float(src.get("vy", 0.0))
    vx2 = safe_float(dst.get("vx", 0.0))
    vy2 = safe_float(dst.get("vy", 0.0))

    heading1 = safe_float(src.get("heading_rad", src.get("heading", 0.0)))
    heading2 = safe_float(dst.get("heading_rad", dst.get("heading", 0.0)))

    cls1 = infer_actor_class(src)
    cls2 = infer_actor_class(dst)

    dx = x2 - x1
    dy = y2 - y1
    distance = math.hypot(dx, dy)

    dvx = vx2 - vx1
    dvy = vy2 - vy1
    relative_speed = math.hypot(dvx, dvy)

    if distance > 1e-8:
        bearing = math.atan2(dy, dx)
        closing_speed = compute_closing_speed(dx, dy, dvx, dvy, distance)
    else:
        bearing = 0.0
        closing_speed = 0.0

    rel_heading = wrap_angle_rad(heading2 - heading1)

    ttc = compute_ttc_simple(
        x1,
        y1,
        vx1,
        vy1,
        x2,
        y2,
        vx2,
        vy2,
        collision_radius=collision_radius,
        max_ttc=max_ttc,
    )

    return {
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "dvx": dvx,
        "dvy": dvy,
        "relative_speed": relative_speed,
        "closing_speed": closing_speed,
        "bearing_sin": math.sin(bearing),
        "bearing_cos": math.cos(bearing),
        "rel_heading_sin": math.sin(rel_heading),
        "rel_heading_cos": math.cos(rel_heading),
        "ttc": ttc,
        "same_class": 1.0 if cls1 == cls2 else 0.0,
        "size_src": float(SIZE_PRIORS.get(cls1, SIZE_PRIORS["other"])),
        "size_dst": float(SIZE_PRIORS.get(cls2, SIZE_PRIORS["other"])),
    }


def reverse_edge_feature(edge_feat: dict[str, float]) -> dict[str, float]:
    return {
        "dx": -float(edge_feat["dx"]),
        "dy": -float(edge_feat["dy"]),
        "distance": float(edge_feat["distance"]),
        "dvx": -float(edge_feat["dvx"]),
        "dvy": -float(edge_feat["dvy"]),
        "relative_speed": float(edge_feat["relative_speed"]),
        "closing_speed": float(edge_feat["closing_speed"]),
        "bearing_sin": -float(edge_feat["bearing_sin"]),
        "bearing_cos": -float(edge_feat["bearing_cos"]),
        "rel_heading_sin": -float(edge_feat["rel_heading_sin"]),
        "rel_heading_cos": float(edge_feat["rel_heading_cos"]),
        "ttc": float(edge_feat["ttc"]),
        "same_class": float(edge_feat["same_class"]),
        "size_src": float(edge_feat["size_dst"]),
        "size_dst": float(edge_feat["size_src"]),
    }


def edge_feature_to_list(edge_feat: dict[str, float]) -> list[float]:
    return [float(edge_feat[name]) for name in EDGE_FEATURE_ORDER]


def edge_feature_dim() -> int:
    return len(EDGE_FEATURE_ORDER)
