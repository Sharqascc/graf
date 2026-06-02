from __future__ import annotations

from typing import Any

import torch
from torch_geometric.data import Data


def _f(d: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = d.get(key, default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _feature_list(node: dict[str, Any]) -> list[float]:
    feats = node.get("features")
    if isinstance(feats, list) and feats:
        out: list[float] = []
        for v in feats:
            try:
                out.append(float(v))
            except (TypeError, ValueError):
                out.append(0.0)
        return out

    class_one_hot = node.get("class_one_hot", [])
    safe_one_hot: list[float] = []
    if isinstance(class_one_hot, list):
        for v in class_one_hot:
            try:
                safe_one_hot.append(float(v))
            except (TypeError, ValueError):
                safe_one_hot.append(0.0)

    return [
        _f(node, "x"),
        _f(node, "y"),
        _f(node, "vx"),
        _f(node, "vy"),
        _f(node, "speed"),
        _f(node, "acceleration", _f(node, "ax")),
        _f(node, "heading"),
        *safe_one_hot,
    ]


def _node_matrix(nodes: list[dict[str, Any]]) -> torch.Tensor:
    rows = [_feature_list(node) for node in nodes]
    if not rows:
        rows = [[0.0]]
    width = max(len(r) for r in rows)
    rows = [r + [0.0] * (width - len(r)) for r in rows]
    return torch.tensor(rows, dtype=torch.float)


def _edge_endpoints(edge: dict[str, Any]) -> tuple[int | None, int | None]:
    if "src_node" in edge and "dst_node" in edge:
        try:
            return int(edge["src_node"]), int(edge["dst_node"])
        except (TypeError, ValueError):
            return None, None

    for s_key, d_key in [("source", "target"), ("src", "dst"), ("u", "v")]:
        if s_key in edge and d_key in edge:
            try:
                return int(edge[s_key]), int(edge[d_key])
            except (TypeError, ValueError):
                return None, None

    return None, None


def _edge_index(edges: list[dict[str, Any]], num_nodes: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    pairs: list[list[int]] = []
    kept_edges: list[dict[str, Any]] = []

    for edge in edges:
        src, dst = _edge_endpoints(edge)
        if src is None or dst is None:
            continue
        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            pairs.append([src, dst])
            kept_edges.append(edge)

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long), []

    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), kept_edges


def _edge_attr(edges: list[dict[str, Any]]) -> torch.Tensor:
    rows: list[list[float]] = []
    for edge in edges:
        rows.append([
            _f(edge, "dx"),
            _f(edge, "dy"),
            _f(edge, "distance"),
            _f(edge, "dvx"),
            _f(edge, "dvy"),
            _f(edge, "relative_speed"),
            _f(edge, "bearing"),
            _f(edge, "heading_delta"),
            1.0 if bool(edge.get("same_class", False)) else 0.0,
        ])

    if not rows:
        return torch.empty((0, 9), dtype=torch.float)

    return torch.tensor(rows, dtype=torch.float)


def to_pyg_dict(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    x = _node_matrix(nodes)
    edge_index, kept_edges = _edge_index(edges, num_nodes=x.size(0))
    edge_attr = _edge_attr(kept_edges)

    payload = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.tensor([_f(graph, "label")], dtype=torch.float),
    }

    for key in ["site_id", "sample_id", "video_id", "window_id", "frame_id"]:
        if key in graph:
            payload[key] = graph[key]

    payload["num_nodes"] = x.size(0)
    return payload


def to_pyg_data(graph: dict[str, Any]) -> Data:
    return Data(**to_pyg_dict(graph))
