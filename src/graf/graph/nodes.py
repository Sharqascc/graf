from __future__ import annotations
from dataclasses import asdict, is_dataclass
from math import atan2, hypot
from typing import Any, Iterable

def _to_mapping(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if is_dataclass(obj):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in vars(obj).items() if not k.startswith("_")}
    raise TypeError(f"Unsupported record type: {type(obj)!r}")

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default

def _one_hot(label: str, classes: list[str]) -> list[float]:
    return [1.0 if label == c else 0.0 for c in classes]

def build_node_feature(record: Any, actor_classes: list[str] | None = None) -> dict[str, Any]:
    row = _to_mapping(record)
    actor_class = str(row.get("actor_class", row.get("class_name", "unknown")))
    x = _safe_float(row.get("x"))
    y = _safe_float(row.get("y"))
    vx = _safe_float(row.get("vx"))
    vy = _safe_float(row.get("vy"))
    ax = _safe_float(row.get("ax"))
    ay = _safe_float(row.get("ay"))
    speed = _safe_float(row.get("speed"), hypot(vx, vy))
    acceleration = _safe_float(row.get("acceleration"), hypot(ax, ay))
    heading = _safe_float(row.get("heading"), atan2(vy, vx) if (vx != 0.0 or vy != 0.0) else 0.0)

    feature = {
        "track_id": row.get("track_id", row.get("id")),
        "frame_id": row.get("frame_id", row.get("frame")),
        "actor_class": actor_class,
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "ax": ax,
        "ay": ay,
        "speed": speed,
        "acceleration": acceleration,
        "heading": heading,
    }

    if actor_classes:
        feature["class_one_hot"] = _one_hot(actor_class, actor_classes)

    return feature

def build_nodes(records: Iterable[Any], actor_classes: list[str] | None = None) -> list[dict[str, Any]]:
    return [build_node_feature(r, actor_classes=actor_classes) for r in records]
