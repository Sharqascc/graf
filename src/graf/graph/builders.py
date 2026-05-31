from __future__ import annotations

import math
from typing import Any

import networkx as nx

# Global actor-class list used everywhere for one-hot encoding
ACTOR_CLASSES = ["car", "pedestrian", "two_wheeler", "auto_rickshaw", "bus", "truck", "bicycle", "other"]

CLASS_TO_INDEX = {name: i for i, name in enumerate(ACTOR_CLASSES)}


def one_hot_actor_class(name: str) -> list[float]:
    idx = CLASS_TO_INDEX.get(name, CLASS_TO_INDEX["other"])
    vec = [0.0] * len(ACTOR_CLASSES)
    vec[idx] = 1.0
    return vec


def safe_float(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def build_graph_for_frame(
    actors: list[dict[str, Any]],
    radius: float = 5.0,
    actor_classes: list[str] | None = None,
    directed: bool = False,
) -> dict[str, Any]:
    """
    Build a graph for a single frame with FIXED feature dimension:

    Node features (per actor), in this exact order:
      [x, y, vx, vy, speed, heading] + one_hot_actor_class

    This guarantees the same number of features for every node in every graph.
    """
    cls_list = actor_classes if actor_classes is not None else ACTOR_CLASSES
    class_set = set(cls_list)

    nodes = []
    for a in actors:
        cls = str(a.get("actor_class", "other"))
        if cls not in class_set:
            continue

        x = safe_float(a.get("x", 0.0))
        y = safe_float(a.get("y", 0.0))
        vx = safe_float(a.get("vx", 0.0))
        vy = safe_float(a.get("vy", 0.0))
        speed = math.sqrt(vx * vx + vy * vy)
        heading = safe_float(a.get("heading", 0.0))

        one_hot = one_hot_actor_class(cls)

        feature_vec = [x, y, vx, vy, speed, heading] + one_hot

        nodes.append({
            "track_id": int(a.get("track_id", 0)),
            "frame_id": int(a.get("frame_id", 0)),
            "actor_class": cls,
            "x": x,
            "y": y,
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "acceleration": safe_float(a.get("acceleration", 0.0)),
            "heading": heading,
            "class_one_hot": one_hot,
            "features": feature_vec,
        })

    G = nx.Graph()
    for i, n in enumerate(nodes):
        G.add_node(i, **n)

    n_nodes = len(nodes)
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            n1 = G.nodes[i]
            n2 = G.nodes[j]

            dx = n2["x"] - n1["x"]
            dy = n2["y"] - n1["y"]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist > radius:
                continue

            dvx = n2["vx"] - n1["vx"]
            dvy = n2["vy"] - n1["vy"]
            rel_speed = math.sqrt(dvx * dvx + dvy * dvy)

            heading_delta = n2["heading"] - n1["heading"]
            same_class = n1["actor_class"] == n2["actor_class"]

            edge_data = {
                "src": n1["track_id"],
                "dst": n2["track_id"],
                "dx": dx,
                "dy": dy,
                "distance": dist,
                "dvx": dvx,
                "dvy": dvy,
                "relative_speed": rel_speed,
                "bearing": math.atan2(dy, dx),
                "heading_delta": heading_delta,
                "same_class": same_class,
            }

            G.add_edge(i, j, **edge_data)

            if directed:
                G.add_edge(j, i, **edge_data)

    out_nodes = []
    for i in G.nodes():
        n = dict(G.nodes[i])
        out_nodes.append(n)

    out_edges = []
    for u, v, e in G.edges(data=True):
        out_edges.append({**e, "src_node": u, "dst_node": v})

    return {
        "frame_id": int(nodes[0]["frame_id"]) if nodes else 0,
        "num_nodes": len(out_nodes),
        "num_edges": len(out_edges),
        "nodes": out_nodes,
        "edges": out_edges,
        "radius": radius,
        "directed": directed,
    }


def build_interaction_graph(
    actors,
    radius: float = 5.0,
    actor_classes=None,
    directed: bool = False,
):
    """Backward-compatible wrapper for older imports."""
    return build_graph_for_frame(
        actors=actors,
        radius=radius,
        actor_classes=actor_classes,
        directed=directed,
    )
