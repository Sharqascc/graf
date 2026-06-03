from __future__ import annotations

from typing import Iterable

import torch
from torch_geometric.data import Data


def _require_attr(data: Data, name: str) -> torch.Tensor:
    value = getattr(data, name, None)
    if value is None:
        raise ValueError(f"Expected Data object to have attribute '{name}'")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Expected '{name}' to be a torch.Tensor, got {type(value)!r}")
    return value


def _frame_sort_key(data: Data) -> tuple[str, int]:
    video_id = str(getattr(data, "video_id", "unknown"))
    frame_id = int(getattr(data, "frame_id", 0))
    return (video_id, frame_id)


def _shift_edge_index(edge_index: torch.Tensor, offset: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return edge_index.clone()
    return edge_index + offset


def build_temporal_window_graph(
    frames: Iterable[Data],
    connect_bidirectional: bool = True,
    temporal_self_only: bool = False,
) -> Data:
    """
    Merge a sequence of per-frame spatial graphs into one spatio-temporal graph.

    Result is a homogeneous PyG Data object with:
      - spatial edges from each frame
      - temporal edges linking same track_id across consecutive frames
      - node metadata describing frame membership
      - edge_type: 0 for spatial, 1 for temporal
    """
    frame_list = sorted(list(frames), key=_frame_sort_key)
    if not frame_list:
        raise ValueError("At least one frame graph is required")

    video_ids = {str(getattr(g, "video_id", "unknown")) for g in frame_list}
    if len(video_ids) > 1:
        raise ValueError("All frames in a temporal window must come from the same video_id")

    x_parts: list[torch.Tensor] = []
    pos_parts: list[torch.Tensor] = []
    spatial_edge_indices: list[torch.Tensor] = []
    spatial_edge_attrs: list[torch.Tensor] = []
    temporal_src: list[int] = []
    temporal_dst: list[int] = []

    track_ids_parts: list[torch.Tensor] = []
    actor_class_index_parts: list[torch.Tensor] = []
    node_frame_index_parts: list[torch.Tensor] = []
    node_frame_id_parts: list[torch.Tensor] = []

    frame_offsets: list[int] = []
    offset = 0

    for t, g in enumerate(frame_list):
        if getattr(g, "x", None) is None:
            raise ValueError("Each frame graph must contain x")
        if getattr(g, "edge_index", None) is None:
            raise ValueError("Each frame graph must contain edge_index")
        if getattr(g, "edge_attr", None) is None:
            raise ValueError("Each frame graph must contain edge_attr")
        if getattr(g, "pos", None) is None:
            raise ValueError("Each frame graph must contain pos")
        if getattr(g, "track_ids", None) is None:
            raise ValueError("Each frame graph must contain track_ids")

        num_nodes = int(g.num_nodes)
        frame_offsets.append(offset)

        x_parts.append(g.x)
        pos_parts.append(g.pos)
        track_ids_parts.append(g.track_ids)
        actor_class_index_parts.append(
            getattr(g, "actor_class_index", torch.full((num_nodes,), -1, dtype=torch.long))
        )
        node_frame_index_parts.append(torch.full((num_nodes,), t, dtype=torch.long))
        node_frame_id_parts.append(torch.full((num_nodes,), int(getattr(g, "frame_id", t)), dtype=torch.long))

        spatial_edge_indices.append(_shift_edge_index(g.edge_index, offset))
        spatial_edge_attrs.append(g.edge_attr)

        offset += num_nodes

    x = torch.cat(x_parts, dim=0)
    pos = torch.cat(pos_parts, dim=0)
    track_ids = torch.cat(track_ids_parts, dim=0)
    actor_class_index = torch.cat(actor_class_index_parts, dim=0)
    node_frame_index = torch.cat(node_frame_index_parts, dim=0)
    node_frame_id = torch.cat(node_frame_id_parts, dim=0)

    for t in range(len(frame_list) - 1):
        g_a = frame_list[t]
        g_b = frame_list[t + 1]
        off_a = frame_offsets[t]
        off_b = frame_offsets[t + 1]

        map_a = {int(tid): i for i, tid in enumerate(g_a.track_ids.tolist())}
        map_b = {int(tid): i for i, tid in enumerate(g_b.track_ids.tolist())}

        shared_ids = sorted(set(map_a).intersection(map_b))
        for tid in shared_ids:
            i = off_a + map_a[tid]
            j = off_b + map_b[tid]

            if temporal_self_only and tid < 0:
                continue

            temporal_src.append(i)
            temporal_dst.append(j)
            if connect_bidirectional:
                temporal_src.append(j)
                temporal_dst.append(i)

    node_feat_dim = int(x.shape[1])
    spatial_edge_feat_dim = 0
    if spatial_edge_attrs:
        spatial_edge_feat_dim = int(spatial_edge_attrs[0].shape[1]) if spatial_edge_attrs[0].numel() else int(
            max((ea.shape[1] for ea in spatial_edge_attrs), default=0)
        )

    spatial_edge_index = (
        torch.cat(spatial_edge_indices, dim=1)
        if spatial_edge_indices
        else torch.empty((2, 0), dtype=torch.long)
    )
    spatial_edge_attr = (
        torch.cat(spatial_edge_attrs, dim=0)
        if spatial_edge_attrs
        else torch.empty((0, spatial_edge_feat_dim), dtype=torch.float32)
    )

    temporal_edge_index = (
        torch.tensor([temporal_src, temporal_dst], dtype=torch.long)
        if temporal_src
        else torch.empty((2, 0), dtype=torch.long)
    )
    temporal_edge_attr = torch.zeros(
        (temporal_edge_index.shape[1], spatial_edge_feat_dim),
        dtype=spatial_edge_attr.dtype if spatial_edge_attr.numel() else torch.float32,
    )

    edge_index = torch.cat([spatial_edge_index, temporal_edge_index], dim=1)
    edge_attr = torch.cat([spatial_edge_attr, temporal_edge_attr], dim=0)

    edge_type_spatial = torch.zeros(spatial_edge_index.shape[1], dtype=torch.long)
    edge_type_temporal = torch.ones(temporal_edge_index.shape[1], dtype=torch.long)
    edge_type = torch.cat([edge_type_spatial, edge_type_temporal], dim=0)

    data = Data(
        x=x,
        pos=pos,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_type=edge_type,
        num_nodes=int(x.shape[0]),
    )

    data.track_ids = track_ids
    data.actor_class_index = actor_class_index
    data.node_frame_index = node_frame_index
    data.node_frame_id = node_frame_id
    data.window_size = len(frame_list)
    data.video_id = str(getattr(frame_list[0], "video_id", "unknown"))
    data.frame_ids = torch.tensor([int(getattr(g, "frame_id", i)) for i, g in enumerate(frame_list)], dtype=torch.long)
    data.num_spatial_edges = int(spatial_edge_index.shape[1])
    data.num_temporal_edges = int(temporal_edge_index.shape[1])

    labels = [getattr(g, "y", None) for g in frame_list if getattr(g, "y", None) is not None]
    if labels:
        try:
            data.y = labels[-1]
        except Exception:
            pass

    return data
