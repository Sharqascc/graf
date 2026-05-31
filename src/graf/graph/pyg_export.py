from __future__ import annotations

from typing import Any


def _contiguous_index(nodes: list[dict[str, Any]]) -> dict[int, int]:
    return {int(node["track_id"]): idx for idx, node in enumerate(nodes)}


def _node_matrix(nodes: list[dict[str, Any]]) -> list[list[float]]:
    rows = []
    for node in nodes:
        rows.append([
            float(node["x"]),
            float(node["y"]),
            float(node["vx"]),
            float(node["vy"]),
            float(node["ax"]),
            float(node["ay"]),
            float(node["speed"]),
            float(node["acceleration"]),
            float(node["heading"]),
            *[float(v) for v in node.get("class_one_hot", [])],
        ])
    return rows


def _edge_index_and_attr(edges: list[dict[str, Any]], node_map: dict[int, int]) -> tuple[list[list[int]], list[list[float]]]:
    edge_index = [[], []]
    edge_attr = []
    for edge in edges:
        src = node_map[int(edge["src"])]
        dst = node_map[int(edge["dst"])]
        edge_index[0].append(src)
        edge_index[1].append(dst)
        edge_attr.append([
            float(edge["dx"]),
            float(edge["dy"]),
            float(edge["distance"]),
            float(edge["dvx"]),
            float(edge["dvy"]),
            float(edge["relative_speed"]),
            float(edge["bearing"]),
            float(edge["heading_delta"]),
            float(edge["same_class"]),
        ])
    return edge_index, edge_attr


def to_pyg_dict(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    node_map = _contiguous_index(nodes)
    x = _node_matrix(nodes)
    edge_index, edge_attr = _edge_index_and_attr(edges, node_map)
    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "num_nodes": len(nodes),
        "frame_id": graph.get("frame_id"),
        "track_ids": [int(n["track_id"]) for n in nodes],
    }


def to_pyg_data(graph: dict[str, Any]) -> Any:
    payload = to_pyg_dict(graph)
    try:
        import torch
        from torch_geometric.data import Data
    except Exception:
        return payload

    x = torch.tensor(payload["x"], dtype=torch.float32)
    edge_index = torch.tensor(payload["edge_index"], dtype=torch.long)
    edge_attr = torch.tensor(payload["edge_attr"], dtype=torch.float32)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = payload["num_nodes"]
    data.frame_id = payload["frame_id"]
    data.track_ids = payload["track_ids"]
    return data
