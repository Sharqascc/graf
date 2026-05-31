from __future__ import annotations
from collections import defaultdict
from typing import Any, Iterable
from .edges import build_edge_feature
from .nodes import build_nodes

def build_graph_for_frame(records: Iterable[Any], radius: float = 15.0, actor_classes: list[str] | None = None, directed: bool = False) -> dict[str, Any]:
    nodes = build_nodes(records, actor_classes=actor_classes)
    edges: list[dict[str, Any]] = []

    for i, src in enumerate(nodes):
        for j, dst in enumerate(nodes):
            if i == j:
                continue
            if not directed and j <= i:
                continue
            edge = build_edge_feature(src, dst)
            if edge["distance"] <= radius:
                edges.append(edge)

    return {
        "frame_id": nodes[0]["frame_id"] if nodes else None,
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "radius": radius,
        "directed": directed,
    }

def build_interaction_graph(records: Iterable[Any], radius: float = 15.0, actor_classes: list[str] | None = None, directed: bool = False) -> list[dict[str, Any]]:
    grouped: dict[Any, list[Any]] = defaultdict(list)
    for record in records:
        frame_id = getattr(record, "frame_id", None)
        if frame_id is None and isinstance(record, dict):
            frame_id = record.get("frame_id", record.get("frame"))
        grouped[frame_id].append(record)

    graphs = []
    for frame_id in sorted(grouped):
        g = build_graph_for_frame(grouped[frame_id], radius=radius, actor_classes=actor_classes, directed=directed)
        g["frame_id"] = frame_id
        graphs.append(g)
    return graphs
