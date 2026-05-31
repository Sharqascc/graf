from __future__ import annotations
from typing import Any

def graph_to_tensors(graph: dict[str, Any]) -> dict[str, Any]:
    node_features = [
        [n["x"], n["y"], n["vx"], n["vy"], n["ax"], n["ay"], n["speed"], n["acceleration"], n["heading"], *n.get("class_one_hot", [])]
        for n in graph.get("nodes", [])
    ]
    edge_index = [[e["src"], e["dst"]] for e in graph.get("edges", [])]
    edge_features = [
        [e["dx"], e["dy"], e["distance"], e["dvx"], e["dvy"], e["relative_speed"], e["bearing"], e["heading_delta"], float(e["same_class"])]
        for e in graph.get("edges", [])
    ]
    return {
        "x": node_features,
        "edge_index": edge_index,
        "edge_attr": edge_features,
        "frame_id": graph.get("frame_id"),
    }
