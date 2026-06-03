from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import torch
from torch_geometric.data import Data

from .builders import GraphBuilder


def package_tensor_graph(
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos: torch.Tensor | None = None,
    y: torch.Tensor | None = None,
    metadata: dict[str, Any] | None = None,
) -> Data:
    """
    Package already-built tensors into a canonical PyG Data object.
    """
    if x.dim() != 2:
        raise ValueError(f"x must be 2D [num_nodes, num_features], got {tuple(x.shape)}")
    if edge_index.dim() != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must be [2, num_edges], got {tuple(edge_index.shape)}")
    if edge_attr.dim() != 2:
        raise ValueError(f"edge_attr must be 2D [num_edges, num_features], got {tuple(edge_attr.shape)}")
    if edge_attr.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_attr rows ({edge_attr.shape[0]}) must equal num_edges ({edge_index.shape[1]})"
        )

    if pos is None:
        if x.shape[1] < 2:
            raise ValueError("pos is None and x has fewer than 2 columns; cannot infer positions")
        pos = x[:, :2]

    if pos.dim() != 2 or pos.shape[1] != 2:
        raise ValueError(f"pos must be [num_nodes, 2], got {tuple(pos.shape)}")
    if pos.shape[0] != x.shape[0]:
        raise ValueError("pos and x must have the same number of nodes")

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        pos=pos,
        num_nodes=int(x.shape[0]),
    )

    if y is not None:
        data.y = y

    if metadata:
        for key, value in metadata.items():
            setattr(data, key, value)

    return data


def save_graph_sample(
    data: Data,
    output_dir: str | Path,
    prefix: str = "sample",
) -> Path:
    """
    Save a single PyG Data graph to disk as a .pt file.
    """
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    video_id = getattr(data, "video_id", "unknown")
    frame_id = getattr(data, "frame_id", "0")

    if isinstance(frame_id, int):
        filename = f"{prefix}_{video_id}_f{frame_id:06d}.pt"
    else:
        filename = f"{prefix}_{video_id}_f{frame_id}.pt"

    path = out_dir / filename
    torch.save(data.to("cpu"), path)
    return path


def load_graph_sample(path: str | Path) -> Data:
    """
    Load a single serialized PyG Data graph from disk.
    """
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"No graph sample found at: {path}")

    data = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(data, Data):
        raise TypeError(f"Expected a PyG Data object in {path}, got {type(data)!r}")
    return data


def to_pyg_data(graph: dict[str, Any]) -> Data:
    """
    Deprecated legacy adapter from dict-graph to canonical PyG Data.
    """
    warnings.warn(
        "graf.graph.pyg_export.to_pyg_data() is deprecated; use GraphBuilder.build_pyg_data() instead.",
        DeprecationWarning,
        stacklevel=2,
    )

    actors = []
    for node in graph.get("nodes", []):
        actors.append(
            {
                "track_id": node.get("track_id"),
                "x_m": node.get("x", node.get("x_m", 0.0)),
                "y_m": node.get("y", node.get("y_m", 0.0)),
                "vx": node.get("vx", 0.0),
                "vy": node.get("vy", 0.0),
                "ax": node.get("ax", 0.0),
                "ay": node.get("ay", 0.0),
                "heading_rad": node.get("heading", 0.0),
                "actor_class": node.get("actor_class", "other"),
                "frame_id": graph.get("frame_id"),
            }
        )

    data = GraphBuilder().build_pyg_data(
        actors=actors,
        frame_id=graph.get("frame_id"),
        video_id=graph.get("video_id"),
    )

    if "label" in graph:
        data.y = torch.tensor([float(graph["label"])], dtype=torch.float32)

    for key in ("site_id", "sample_id", "window_id"):
        if key in graph:
            setattr(data, key, graph[key])

    return data
