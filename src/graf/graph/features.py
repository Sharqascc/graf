from __future__ import annotations

from typing import Final

import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph

from .edges import EDGE_FEATURE_ORDER

NODE_BASE_FEATURE_ORDER: Final[list[str]] = [
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "ax",
    "ay",
    "accel_mag",
    "sin_heading",
    "cos_heading",
    "size_prior",
]

EDGE_KINEMATIC_FEATURE_ORDER: Final[list[str]] = [
    "dx",
    "dy",
    "distance",
    "dvx",
    "dvy",
    "relative_speed",
    "closing_speed",
    "ttc",
]

EDGE_ANGULAR_FEATURE_ORDER: Final[list[str]] = [
    "bearing_sin",
    "bearing_cos",
    "rel_heading_sin",
    "rel_heading_cos",
]


def num_node_base_features() -> int:
    return len(NODE_BASE_FEATURE_ORDER)


def num_edge_features() -> int:
    return len(EDGE_FEATURE_ORDER)


def extract_node_base_features(data: Data) -> torch.Tensor:
    """
    Return the continuous, non-class node channels.

    Layout:
      [x, y, vx, vy, speed, ax, ay, accel_mag, sin_heading, cos_heading, size_prior]
    """
    width = num_node_base_features()
    if getattr(data, "x", None) is None:
        return torch.empty((0, width), dtype=torch.float32)
    if data.x.numel() == 0:
        return torch.empty((0, width), dtype=data.x.dtype, device=data.x.device)
    if data.x.size(1) < width:
        raise ValueError(
            f"Expected at least {width} node features, got {data.x.size(1)}."
        )
    return data.x[:, :width]


def extract_kinematic_features(data: Data) -> torch.Tensor:
    """
    Return the core kinematic node channels.

    Layout:
      [x, y, vx, vy, speed, ax, ay, accel_mag]
    """
    if getattr(data, "x", None) is None:
        return torch.empty((0, 8), dtype=torch.float32)
    if data.x.numel() == 0:
        return torch.empty((0, 8), dtype=data.x.dtype, device=data.x.device)
    if data.x.size(1) < 8:
        raise ValueError(
            f"Expected at least 8 node features, got {data.x.size(1)}."
        )
    return data.x[:, :8]


def extract_heading_features(data: Data) -> torch.Tensor:
    """
    Return [sin_heading, cos_heading].
    """
    if getattr(data, "x", None) is None:
        return torch.empty((0, 2), dtype=torch.float32)
    if data.x.numel() == 0:
        return torch.empty((0, 2), dtype=data.x.dtype, device=data.x.device)
    if data.x.size(1) < 10:
        raise ValueError(
            f"Expected at least 10 node features, got {data.x.size(1)}."
        )
    return data.x[:, 8:10]


def extract_class_one_hot(data: Data) -> torch.Tensor:
    """
    Return the one-hot class block from node features.
    """
    if getattr(data, "x", None) is None:
        return torch.empty((0, 0), dtype=torch.float32)
    if data.x.numel() == 0:
        width = max(data.x.size(1) - num_node_base_features(), 0)
        return torch.empty((0, width), dtype=data.x.dtype, device=data.x.device)
    if data.x.size(1) <= num_node_base_features():
        return torch.empty((data.x.size(0), 0), dtype=data.x.dtype, device=data.x.device)
    return data.x[:, num_node_base_features():]


def extract_edge_distance(data: Data) -> torch.Tensor:
    """
    Return the edge distance column as shape [E, 1].
    """
    if getattr(data, "edge_attr", None) is None:
        return torch.empty((0, 1), dtype=torch.float32)
    if data.edge_attr.numel() == 0:
        return torch.empty((0, 1), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
    if data.edge_attr.size(1) < 3:
        raise ValueError(
            f"Expected at least 3 edge features, got {data.edge_attr.size(1)}."
        )
    return data.edge_attr[:, 2:3]


def extract_edge_kinematics(data: Data) -> torch.Tensor:
    """
    Return core edge motion channels.

    Layout:
      [dx, dy, distance, dvx, dvy, relative_speed, closing_speed, ttc]
    """
    if getattr(data, "edge_attr", None) is None:
        return torch.empty((0, 8), dtype=torch.float32)
    if data.edge_attr.numel() == 0:
        return torch.empty((0, 8), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
    if data.edge_attr.size(1) < len(EDGE_FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(EDGE_FEATURE_ORDER)} edge features, got {data.edge_attr.size(1)}."
        )
    idx = [0, 1, 2, 3, 4, 5, 6, 11]
    return data.edge_attr[:, idx]


def extract_edge_angles(data: Data) -> torch.Tensor:
    """
    Return angular edge channels.

    Layout:
      [bearing_sin, bearing_cos, rel_heading_sin, rel_heading_cos]
    """
    if getattr(data, "edge_attr", None) is None:
        return torch.empty((0, 4), dtype=torch.float32)
    if data.edge_attr.numel() == 0:
        return torch.empty((0, 4), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
    if data.edge_attr.size(1) < len(EDGE_FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(EDGE_FEATURE_ORDER)} edge features, got {data.edge_attr.size(1)}."
        )
    idx = [7, 8, 9, 10]
    return data.edge_attr[:, idx]


def extract_edge_risk_features(data: Data) -> torch.Tensor:
    """
    Return edge channels most directly tied to interaction risk.

    Layout:
      [distance, relative_speed, closing_speed, ttc, same_class, size_src, size_dst]
    """
    if getattr(data, "edge_attr", None) is None:
        return torch.empty((0, 7), dtype=torch.float32)
    if data.edge_attr.numel() == 0:
        return torch.empty((0, 7), dtype=data.edge_attr.dtype, device=data.edge_attr.device)
    if data.edge_attr.size(1) < len(EDGE_FEATURE_ORDER):
        raise ValueError(
            f"Expected {len(EDGE_FEATURE_ORDER)} edge features, got {data.edge_attr.size(1)}."
        )
    idx = [2, 5, 6, 11, 12, 13, 14]
    return data.edge_attr[:, idx]


def filter_subgraph_by_class(
    data: Data,
    target_class_idx: int,
    *,
    keep_isolated_nodes: bool = True,
) -> Data:
    """
    Extract the induced subgraph for nodes whose actor_class_index matches target_class_idx.

    If keep_isolated_nodes is True, matching nodes are kept even if they end up with no edges.
    """
    if not hasattr(data, "actor_class_index"):
        raise AttributeError("Data object is missing required attribute 'actor_class_index'.")

    if data.num_nodes == 0:
        return data.clone()

    node_mask = data.actor_class_index == int(target_class_idx)
    subset = node_mask.nonzero(as_tuple=True)[0]

    return filter_subgraph_by_node_mask(
        data,
        node_mask=node_mask,
        keep_isolated_nodes=keep_isolated_nodes,
        subset=subset,
    )


def filter_subgraph_by_node_mask(
    data: Data,
    node_mask: torch.Tensor,
    *,
    keep_isolated_nodes: bool = True,
    subset: torch.Tensor | None = None,
) -> Data:
    """
    Extract an induced subgraph from a boolean node mask.

    Uses PyG's subgraph utility to keep edge_index and edge_attr aligned and relabeled.
    """
    if node_mask.dtype != torch.bool:
        raise TypeError("node_mask must be a boolean tensor.")
    if node_mask.dim() != 1:
        raise ValueError("node_mask must be one-dimensional.")
    if node_mask.numel() != data.num_nodes:
        raise ValueError(
            f"node_mask length {node_mask.numel()} does not match num_nodes {data.num_nodes}."
        )

    if subset is None:
        subset = node_mask.nonzero(as_tuple=True)[0]

    if subset.numel() == 0:
        return _empty_like(data)

    edge_index, edge_attr = subgraph(
        subset,
        data.edge_index,
        data.edge_attr if hasattr(data, "edge_attr") else None,
        relabel_nodes=True,
        num_nodes=data.num_nodes,
        return_edge_mask=False,
    )

    if keep_isolated_nodes:
        x = data.x[node_mask] if hasattr(data, "x") else None
        pos = data.pos[node_mask] if hasattr(data, "pos") else None
        actor_class_index = data.actor_class_index[node_mask] if hasattr(data, "actor_class_index") else None
        track_ids = data.track_ids[node_mask] if hasattr(data, "track_ids") else None
        num_nodes = int(subset.numel())
    else:
        if edge_index.numel() == 0:
            return _empty_like(data)
        kept = torch.unique(edge_index.view(-1), sorted=True)
        x = data.x[subset][kept] if hasattr(data, "x") else None
        pos = data.pos[subset][kept] if hasattr(data, "pos") else None
        actor_class_index = data.actor_class_index[subset][kept] if hasattr(data, "actor_class_index") else None
        track_ids = data.track_ids[subset][kept] if hasattr(data, "track_ids") else None

        remap = torch.full((subset.numel(),), -1, dtype=torch.long, device=edge_index.device)
        remap[kept] = torch.arange(kept.numel(), device=edge_index.device)
        edge_index = remap[edge_index]
        num_nodes = int(kept.numel())

    out = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        num_nodes=num_nodes,
    )

    if pos is not None:
        out.pos = pos
    if actor_class_index is not None:
        out.actor_class_index = actor_class_index
    if track_ids is not None:
        out.track_ids = track_ids
    if hasattr(data, "frame_id"):
        out.frame_id = data.frame_id
    if hasattr(data, "video_id"):
        out.video_id = data.video_id

    return out


def validate_feature_layout(data: Data) -> None:
    """
    Raise an error if the Data object does not match the expected feature layout.
    """
    if getattr(data, "x", None) is None:
        raise ValueError("Data.x is missing.")
    if getattr(data, "edge_index", None) is None:
        raise ValueError("Data.edge_index is missing.")
    if getattr(data, "edge_attr", None) is None:
        raise ValueError("Data.edge_attr is missing.")

    if data.x.dim() != 2:
        raise ValueError(f"Data.x must be rank-2, got shape {tuple(data.x.shape)}.")
    if data.edge_index.dim() != 2 or data.edge_index.size(0) != 2:
        raise ValueError(
            f"Data.edge_index must have shape [2, E], got {tuple(data.edge_index.shape)}."
        )
    if data.edge_attr.dim() != 2:
        raise ValueError(
            f"Data.edge_attr must be rank-2, got shape {tuple(data.edge_attr.shape)}."
        )
    if data.edge_attr.size(0) != data.edge_index.size(1):
        raise ValueError(
            "edge_attr row count must match number of edges: "
            f"{data.edge_attr.size(0)} != {data.edge_index.size(1)}."
        )

    expected_node_min = num_node_base_features()
    expected_edge = len(EDGE_FEATURE_ORDER)

    if data.x.size(1) < expected_node_min:
        raise ValueError(
            f"Expected at least {expected_node_min} node features, got {data.x.size(1)}."
        )
    if data.edge_attr.size(1) != expected_edge:
        raise ValueError(
            f"Expected {expected_edge} edge features, got {data.edge_attr.size(1)}."
        )


def _empty_like(data: Data) -> Data:
    x_width = data.x.size(1) if hasattr(data, "x") and data.x.dim() == 2 else 0
    e_width = data.edge_attr.size(1) if hasattr(data, "edge_attr") and data.edge_attr.dim() == 2 else 0

    x = torch.empty(
        (0, x_width),
        dtype=data.x.dtype if hasattr(data, "x") else torch.float32,
        device=data.x.device if hasattr(data, "x") else None,
    )
    edge_index = torch.empty(
        (2, 0),
        dtype=torch.long,
        device=data.edge_index.device if hasattr(data, "edge_index") else None,
    )
    edge_attr = torch.empty(
        (0, e_width),
        dtype=data.edge_attr.dtype if hasattr(data, "edge_attr") else torch.float32,
        device=data.edge_attr.device if hasattr(data, "edge_attr") else None,
    )

    out = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, num_nodes=0)

    if hasattr(data, "pos"):
        out.pos = torch.empty(
            (0, data.pos.size(1)),
            dtype=data.pos.dtype,
            device=data.pos.device,
        )
    if hasattr(data, "actor_class_index"):
        out.actor_class_index = torch.empty(
            (0,),
            dtype=data.actor_class_index.dtype,
            device=data.actor_class_index.device,
        )
    if hasattr(data, "track_ids"):
        out.track_ids = torch.empty(
            (0,),
            dtype=data.track_ids.dtype,
            device=data.track_ids.device,
        )
    if hasattr(data, "frame_id"):
        out.frame_id = data.frame_id
    if hasattr(data, "video_id"):
        out.video_id = data.video_id

    return out
