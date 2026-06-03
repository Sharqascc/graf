
from __future__ import annotations

from typing import Final, Sequence

import torch
from torch_geometric.data import Data

from .edges import ACTOR_CLASSES

BASE_NODE_FEATURES: Final[tuple[str, ...]] = (
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
)

NODE_FEATURE_MAPPING: Final[dict[str, int]] = {
    name: idx for idx, name in enumerate(BASE_NODE_FEATURES)
}

KINEMATIC_FEATURES: Final[tuple[str, ...]] = (
    "vx",
    "vy",
    "speed",
    "ax",
    "ay",
    "accel_mag",
    "sin_heading",
    "cos_heading",
)

POSITION_FEATURES: Final[tuple[str, ...]] = ("x", "y")


def node_base_feature_dim() -> int:
    return len(BASE_NODE_FEATURES)


def node_feature_dim(actor_classes: Sequence[str] | None = None) -> int:
    classes = tuple(actor_classes) if actor_classes is not None else ACTOR_CLASSES
    return len(BASE_NODE_FEATURES) + len(classes)


def class_feature_offset() -> int:
    return len(BASE_NODE_FEATURES)


def class_feature_slice(actor_classes: Sequence[str] | None = None) -> slice:
    classes = tuple(actor_classes) if actor_classes is not None else ACTOR_CLASSES
    start = class_feature_offset()
    return slice(start, start + len(classes))


def validate_node_layout(
    data: Data,
    actor_classes: Sequence[str] | None = None,
    require_pos: bool = False,
) -> None:
    classes = tuple(actor_classes) if actor_classes is not None else ACTOR_CLASSES

    if not hasattr(data, "x") or data.x is None:
        raise ValueError("Graph data must contain node feature matrix 'x'.")

    if data.x.dim() != 2:
        raise ValueError(f"Expected data.x to be 2D, got shape {tuple(data.x.shape)}.")

    expected_dim = node_feature_dim(classes)
    if data.x.shape[1] != expected_dim:
        raise ValueError(
            f"Unexpected node feature dimension: got {data.x.shape[1]}, "
            f"expected {expected_dim} = {len(BASE_NODE_FEATURES)} base + {len(classes)} class one-hot."
        )

    if require_pos:
        if not hasattr(data, "pos") or data.pos is None:
            raise ValueError("Graph data must contain 'pos' when require_pos=True.")
        if data.pos.dim() != 2 or data.pos.shape[1] != 2:
            raise ValueError(f"Expected data.pos shape [num_nodes, 2], got {tuple(data.pos.shape)}.")

    if hasattr(data, "pos") and data.pos is not None and data.pos.numel() > 0:
        if data.pos.shape[0] != data.x.shape[0]:
            raise ValueError(
                f"Mismatch between x and pos node counts: {data.x.shape[0]} vs {data.pos.shape[0]}."
            )

    if hasattr(data, "actor_class_index") and data.actor_class_index is not None:
        if data.actor_class_index.shape[0] != data.x.shape[0]:
            raise ValueError(
                "Mismatch between actor_class_index length and number of nodes."
            )


def get_node_feature_index(feature_name: str) -> int:
    if feature_name not in NODE_FEATURE_MAPPING:
        valid = ", ".join(NODE_FEATURE_MAPPING.keys())
        raise KeyError(f"Unknown node feature '{feature_name}'. Valid features: {valid}")
    return NODE_FEATURE_MAPPING[feature_name]


def get_node_feature_slice(data: Data, feature_name: str) -> torch.Tensor:
    validate_node_layout(data)
    idx = get_node_feature_index(feature_name)
    if data.x.numel() == 0:
        return torch.empty((0,), dtype=data.x.dtype, device=data.x.device)
    return data.x[:, idx]


def get_node_feature_block(data: Data, feature_names: Sequence[str]) -> torch.Tensor:
    validate_node_layout(data)
    indices = [get_node_feature_index(name) for name in feature_names]
    if data.x.numel() == 0:
        return torch.empty((0, len(indices)), dtype=data.x.dtype, device=data.x.device)
    return data.x[:, indices]


def get_position_features(data: Data) -> torch.Tensor:
    return get_node_feature_block(data, POSITION_FEATURES)


def get_kinematic_features(data: Data) -> torch.Tensor:
    return get_node_feature_block(data, KINEMATIC_FEATURES)


def get_heading_components(data: Data) -> torch.Tensor:
    return get_node_feature_block(data, ("sin_heading", "cos_heading"))


def get_class_one_hot(data: Data, actor_classes: Sequence[str] | None = None) -> torch.Tensor:
    validate_node_layout(data, actor_classes=actor_classes)
    sl = class_feature_slice(actor_classes)
    if data.x.numel() == 0:
        classes = tuple(actor_classes) if actor_classes is not None else ACTOR_CLASSES
        return torch.empty((0, len(classes)), dtype=data.x.dtype, device=data.x.device)
    return data.x[:, sl]


def get_spatial_positions(data: Data) -> torch.Tensor:
    validate_node_layout(data)
    if hasattr(data, "pos") and data.pos is not None and data.pos.numel() > 0:
        return data.pos
    return get_position_features(data)


def get_track_ids(data: Data) -> torch.Tensor | None:
    value = getattr(data, "track_ids", None)
    if value is None:
        return None
    return value


def get_actor_class_index(data: Data) -> torch.Tensor | None:
    value = getattr(data, "actor_class_index", None)
    if value is None:
        return None
    return value


def get_actor_class_names(
    data: Data,
    actor_classes: Sequence[str] | None = None,
) -> list[str]:
    classes = list(actor_classes) if actor_classes is not None else list(ACTOR_CLASSES)
    class_index = get_actor_class_index(data)
    if class_index is None:
        one_hot = get_class_one_hot(data, actor_classes=classes)
        if one_hot.numel() == 0:
            return []
        class_index = one_hot.argmax(dim=1)

    names: list[str] = []
    fallback = "other" if "other" in classes else classes[-1]
    for idx in class_index.detach().cpu().tolist():
        if 0 <= idx < len(classes):
            names.append(classes[idx])
        else:
            names.append(fallback)
    return names


def clone_with_updated_node_feature(
    data: Data,
    feature_name: str,
    values: torch.Tensor,
) -> Data:
    validate_node_layout(data)
    idx = get_node_feature_index(feature_name)

    if values.dim() != 1:
        raise ValueError(
            f"Expected replacement values to be 1D with shape [num_nodes], got {tuple(values.shape)}."
        )
    if values.shape[0] != data.x.shape[0]:
        raise ValueError(
            f"Replacement feature length {values.shape[0]} does not match num_nodes {data.x.shape[0]}."
        )

    out = data.clone()
    out.x = out.x.clone()
    out.x[:, idx] = values.to(device=out.x.device, dtype=out.x.dtype)
    return out


def clone_with_updated_node_feature_block(
    data: Data,
    feature_names: Sequence[str],
    values: torch.Tensor,
) -> Data:
    validate_node_layout(data)
    indices = [get_node_feature_index(name) for name in feature_names]

    if values.dim() != 2:
        raise ValueError(
            f"Expected replacement block to be 2D with shape [num_nodes, num_features], got {tuple(values.shape)}."
        )
    if values.shape[0] != data.x.shape[0]:
        raise ValueError(
            f"Replacement block node count {values.shape[0]} does not match num_nodes {data.x.shape[0]}."
        )
    if values.shape[1] != len(indices):
        raise ValueError(
            f"Replacement block feature count {values.shape[1]} does not match requested feature count {len(indices)}."
        )

    out = data.clone()
    out.x = out.x.clone()
    out.x[:, indices] = values.to(device=out.x.device, dtype=out.x.dtype)
    return out


__all__ = [
    "BASE_NODE_FEATURES",
    "NODE_FEATURE_MAPPING",
    "KINEMATIC_FEATURES",
    "POSITION_FEATURES",
    "node_base_feature_dim",
    "node_feature_dim",
    "class_feature_offset",
    "class_feature_slice",
    "validate_node_layout",
    "get_node_feature_index",
    "get_node_feature_slice",
    "get_node_feature_block",
    "get_position_features",
    "get_kinematic_features",
    "get_heading_components",
    "get_class_one_hot",
    "get_spatial_positions",
    "get_track_ids",
    "get_actor_class_index",
    "get_actor_class_names",
    "clone_with_updated_node_feature",
    "clone_with_updated_node_feature_block",
]
