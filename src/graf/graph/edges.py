from __future__ import annotations
from math import atan2, hypot
from typing import Any

def _rel_heading(a: float, b: float) -> float:
    d = a - b
    while d > 3.141592653589793:
        d -= 2 * 3.141592653589793
    while d < -3.141592653589793:
        d += 2 * 3.141592653589793
    return d

def build_edge_feature(src: dict[str, Any], dst: dict[str, Any]) -> dict[str, float | int | bool]:
    dx = float(dst["x"]) - float(src["x"])
    dy = float(dst["y"]) - float(src["y"])
    dvx = float(dst["vx"]) - float(src["vx"])
    dvy = float(dst["vy"]) - float(src["vy"])
    distance = hypot(dx, dy)
    rel_speed = hypot(dvx, dvy)
    bearing = atan2(dy, dx) if distance > 0 else 0.0
    heading_delta = _rel_heading(float(dst["heading"]), float(src["heading"]))

    return {
        "src": int(src["track_id"]),
        "dst": int(dst["track_id"]),
        "dx": dx,
        "dy": dy,
        "distance": distance,
        "dvx": dvx,
        "dvy": dvy,
        "relative_speed": rel_speed,
        "bearing": bearing,
        "heading_delta": heading_delta,
        "same_class": src.get("actor_class") == dst.get("actor_class"),
    }
