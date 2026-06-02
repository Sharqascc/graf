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


def _node_id(node: dict[str, Any], fallback: int) -> str:
    return str(node.get("id", node.get("node_id", fallback)))


def _node_matrix(nodes: list[dict[str, Any]]) -> tuple[torch.Tensor, dict[str, int]]:
    rows: list[list[float]] = []
    id_map: dict[str, int] = {}

    for idx, node in enumerate(nodes):
        id_map[_node_id(node, idx)] = idx
        rows.append([
            _f(node, "x"),
            _f(node, "y"),
            _f(node, "vx"),
            _f(node, "vy"),
            _f(node, "speed"),
            _f(node, "ax"),
            _f(node, "ay"),
            _f(node, "heading"),
            _f(node, "size"),
            _f(node, "risk_score"),
        ])

    if not rows:
        rows = [[0.0] * 10]

    return torch.tensor(rows, dtype=torch.float), id_map


def _edge_endpoints(edge: dict[str, Any]) -> tuple[Any, Any]:
    src = edge.get("source", edge.get("src", edge.get("u")))
    dst = edge.get("target", edge.get("dst", edge.get("v")))
    return src, dst


def _edge_index(edges: list[dict[str, Any]], id_map: dict[str, int], num_nodes: int) -> torch.Tensor:
    pairs: list[list[int]] = []

    for edge in edges:
        raw_src, raw_dst = _edge_endpoints(edge)

        if raw_src is None or raw_dst is None:
            continue

        src_key = str(raw_src)
        dst_key = str(raw_dst)

        if src_key in id_map and dst_key in id_map:
            src = id_map[src_key]
            dst = id_map[dst_key]
        else:
            try:
                src = int(raw_src)
                dst = int(raw_dst)
            except (TypeError, ValueError):
                continue

        if 0 <= src < num_nodes and 0 <= dst < num_nodes:
            pairs.append([src, dst])

    if not pairs:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.tensor(pairs, dtype=torch.long).t().contiguous()


def _edge_attr(edges: list[dict[str, Any]], edge_count: int) -> torch.Tensor:
    rows: list[list[float]] = []

    for edge in edges:
        rows.append([
            _f(edge, "distance"),
            _f(edge, "ttc"),
            _f(edge, "pet"),
            _f(edge, "risk_score"),
        ])

    if not rows:
        return torch.empty((0, 4), dtype=torch.float)

    rows = rows[:edge_count]
    return torch.tensor(rows, dtype=torch.float)


def to_pyg_dict(graph: dict[str, Any]) -> dict[str, Any]:
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])

    x, id_map = _node_matrix(nodes)
    edge_index = _edge_index(edges, id_map=id_map, num_nodes=x.size(0))
    edge_attr = _edge_attr(edges, edge_count=edge_index.size(1))

    payload = {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "y": torch.tensor([_f(graph, "label")], dtype=torch.float),
    }

    if "site_id" in graph:
        payload["site_id"] = graph["site_id"]
    if "sample_id" in graph:
        payload["sample_id"] = graph["sample_id"]

    return payload


def to_pyg_data(graph: dict[str, Any]) -> Data:
    return Data(**to_pyg_dict(graph))
