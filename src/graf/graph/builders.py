from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch_geometric.data import Data

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


ACTOR_CLASSES = [
    "car",
    "pedestrian",
    "two_wheeler",
    "auto_rickshaw",
    "bus",
    "truck",
    "bicycle",
    "other",
]

CLASS_TO_INDEX = {name: i for i, name in enumerate(ACTOR_CLASSES)}

SIZE_PRIORS = {
    "pedestrian": 0.10,
    "bicycle": 0.20,
    "two_wheeler": 0.25,
    "auto_rickshaw": 0.40,
    "car": 0.60,
    "truck": 0.90,
    "bus": 1.00,
    "other": 0.30,
}

DEFAULT_INTERACTION_RADII = {
    ("car", "car"): 6.0,
    ("car", "pedestrian"): 4.0,
    ("car", "two_wheeler"): 5.0,
    ("car", "auto_rickshaw"): 5.0,
    ("car", "bus"): 7.0,
    ("car", "truck"): 7.0,
    ("car", "bicycle"): 4.0,
    ("pedestrian", "pedestrian"): 2.0,
    ("pedestrian", "two_wheeler"): 3.5,
    ("pedestrian", "auto_rickshaw"): 4.0,
    ("pedestrian", "bus"): 5.0,
    ("pedestrian", "truck"): 5.0,
    ("pedestrian", "bicycle"): 2.5,
    ("two_wheeler", "two_wheeler"): 4.0,
    ("two_wheeler", "auto_rickshaw"): 4.5,
    ("two_wheeler", "bus"): 6.0,
    ("two_wheeler", "truck"): 6.0,
    ("two_wheeler", "bicycle"): 3.0,
    ("auto_rickshaw", "auto_rickshaw"): 5.0,
    ("auto_rickshaw", "bus"): 6.0,
    ("auto_rickshaw", "truck"): 6.0,
    ("auto_rickshaw", "bicycle"): 3.5,
    ("bus", "bus"): 8.0,
    ("bus", "truck"): 8.0,
    ("bus", "bicycle"): 4.5,
    ("truck", "truck"): 8.0,
    ("truck", "bicycle"): 4.5,
    ("bicycle", "bicycle"): 2.5,
}


@dataclass(slots=True)
class FeatureStats:
    node_mean: np.ndarray | None = None
    node_std: np.ndarray | None = None
    edge_mean: np.ndarray | None = None
    edge_std: np.ndarray | None = None


def safe_float(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(out) or math.isinf(out):
        return default
    return out


def wrap_angle_rad(theta: float) -> float:
    return math.atan2(math.sin(theta), math.cos(theta))


def one_hot_actor_class(name: str, actor_classes: list[str] | None = None) -> list[float]:
    cls_list = actor_classes if actor_classes is not None else ACTOR_CLASSES
    cls_to_idx = {n: i for i, n in enumerate(cls_list)}
    idx = cls_to_idx.get(name, cls_to_idx.get("other", len(cls_list) - 1))
    vec = [0.0] * len(cls_list)
    vec[idx] = 1.0
    return vec


def infer_actor_class(actor: dict[str, Any]) -> str:
    raw = (
        actor.get("actor_class")
        or actor.get("class_name")
        or actor.get("class")
        or actor.get("label")
        or "other"
    )
    raw = str(raw).strip().lower()
    return raw if raw in CLASS_TO_INDEX else "other"


def get_pair_radius(
    class_a: str,
    class_b: str,
    default_radius: float,
    pair_radii: dict[tuple[str, str], float] | None = None,
) -> float:
    radii = pair_radii if pair_radii is not None else DEFAULT_INTERACTION_RADII
    if (class_a, class_b) in radii:
        return radii[(class_a, class_b)]
    if (class_b, class_a) in radii:
        return radii[(class_b, class_a)]
    return default_radius


def compute_ttc_simple(
    x1: float,
    y1: float,
    vx1: float,
    vy1: float,
    x2: float,
    y2: float,
    vx2: float,
    vy2: float,
    collision_radius: float = 1.5,
    max_ttc: float = 10.0,
) -> float:
    rx = x2 - x1
    ry = y2 - y1
    rvx = vx2 - vx1
    rvy = vy2 - vy1

    a = rvx * rvx + rvy * rvy
    if a < 1e-8:
        return max_ttc

    b = 2.0 * (rx * rvx + ry * rvy)
    c = rx * rx + ry * ry - collision_radius * collision_radius

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return max_ttc

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    candidates = [t for t in (t1, t2) if 0.0 <= t <= max_ttc]
    if not candidates:
        return max_ttc
    return min(candidates)


class GraphBuilder:
    """
    Research-oriented single-frame traffic interaction graph builder.

    Node feature order:
      [x, y, vx, vy, speed, ax, ay, accel_mag, sin_heading, cos_heading, size_prior] + one_hot_class

    Edge feature order:
      [
        dx, dy, distance,
        dvx, dvy, relative_speed,
        closing_speed,
        bearing_sin, bearing_cos,
        rel_heading_sin, rel_heading_cos,
        ttc_clipped,
        same_class,
        size_src, size_dst
      ]
    """

    def __init__(
        self,
        radius: float = 6.0,
        directed: bool = True,
        actor_classes: list[str] | None = None,
        pair_radii: dict[tuple[str, str], float] | None = None,
        use_kdtree: bool = True,
        use_class_specific_radii: bool = True,
        use_velocity_adaptive_radius: bool = False,
        velocity_radius_scale: float = 0.5,
        min_closing_speed: float | None = None,
        max_ttc: float | None = None,
        collision_radius: float = 1.5,
        normalize: bool = False,
        stats: FeatureStats | None = None,
        include_self_loops: bool = False,
    ) -> None:
        self.radius = radius
        self.directed = directed
        self.actor_classes = actor_classes if actor_classes is not None else ACTOR_CLASSES
        self.pair_radii = pair_radii if pair_radii is not None else DEFAULT_INTERACTION_RADII
        self.use_kdtree = use_kdtree
        self.use_class_specific_radii = use_class_specific_radii
        self.use_velocity_adaptive_radius = use_velocity_adaptive_radius
        self.velocity_radius_scale = velocity_radius_scale
        self.min_closing_speed = min_closing_speed
        self.max_ttc = max_ttc
        self.collision_radius = collision_radius
        self.normalize = normalize
        self.stats = stats
        self.include_self_loops = include_self_loops

    def build_pyg_data(
        self,
        actors: list[dict[str, Any]],
        frame_id: int | None = None,
        video_id: str | None = None,
    ) -> Data:
        if not actors:
            return Data(
                x=torch.empty((0, 11 + len(self.actor_classes)), dtype=torch.float32),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 15), dtype=torch.float32),
                pos=torch.empty((0, 2), dtype=torch.float32),
                num_nodes=0,
            )

        node_rows: list[list[float]] = []
        coords: list[list[float]] = []
        vx_list: list[float] = []
        vy_list: list[float] = []
        heading_list: list[float] = []
        class_list: list[str] = []
        track_ids: list[int] = []
        frame_ids: list[int] = []

        for actor in actors:
            cls = infer_actor_class(actor)

            x = safe_float(actor.get("x_m", actor.get("x", 0.0)))
            y = safe_float(actor.get("y_m", actor.get("y", 0.0)))
            vx = safe_float(actor.get("vx", 0.0))
            vy = safe_float(actor.get("vy", 0.0))
            ax = safe_float(actor.get("ax", 0.0))
            ay = safe_float(actor.get("ay", 0.0))

            speed = math.hypot(vx, vy)
            accel_mag = math.hypot(ax, ay)

            heading = actor.get("heading_rad", actor.get("heading", 0.0))
            heading = safe_float(heading)
            sin_heading = math.sin(heading)
            cos_heading = math.cos(heading)

            size_prior = SIZE_PRIORS.get(cls, SIZE_PRIORS["other"])
            one_hot = one_hot_actor_class(cls, self.actor_classes)

            feat = [
                x,
                y,
                vx,
                vy,
                speed,
                ax,
                ay,
                accel_mag,
                sin_heading,
                cos_heading,
                size_prior,
                *one_hot,
            ]
            node_rows.append(feat)
            coords.append([x, y])
            vx_list.append(vx)
            vy_list.append(vy)
            heading_list.append(heading)
            class_list.append(cls)
            track_ids.append(int(safe_float(actor.get("track_id", len(track_ids)))))
            frame_ids.append(int(safe_float(actor.get("frame_id", frame_id or 0))))

        x_np = np.asarray(node_rows, dtype=np.float32)
        pos_np = np.asarray(coords, dtype=np.float32)
        vx_np = np.asarray(vx_list, dtype=np.float32)
        vy_np = np.asarray(vy_list, dtype=np.float32)
        heading_np = np.asarray(heading_list, dtype=np.float32)
        speed_np = np.sqrt(vx_np * vx_np + vy_np * vy_np).astype(np.float32)
        size_np = np.asarray([SIZE_PRIORS.get(c, SIZE_PRIORS["other"]) for c in class_list], dtype=np.float32)

        edge_pairs = self._find_candidate_pairs(pos_np)

        src_nodes: list[int] = []
        dst_nodes: list[int] = []
        edge_rows: list[list[float]] = []

        for i, j in edge_pairs:
            if i == j and not self.include_self_loops:
                continue

            pair_radius = self.radius
            if self.use_class_specific_radii:
                pair_radius = get_pair_radius(class_list[i], class_list[j], self.radius, self.pair_radii)

            if self.use_velocity_adaptive_radius:
                pair_radius += self.velocity_radius_scale * max(float(speed_np[i]), float(speed_np[j]))

            dx = float(pos_np[j, 0] - pos_np[i, 0])
            dy = float(pos_np[j, 1] - pos_np[i, 1])
            dist = math.hypot(dx, dy)

            if dist <= 1e-8 or dist > pair_radius:
                continue

            dvx = float(vx_np[j] - vx_np[i])
            dvy = float(vy_np[j] - vy_np[i])
            rel_speed = math.hypot(dvx, dvy)

            ux = dx / dist
            uy = dy / dist
            closing_speed = -(dvx * ux + dvy * uy)

            if self.min_closing_speed is not None and closing_speed < self.min_closing_speed:
                continue

            bearing = math.atan2(dy, dx)
            rel_heading = wrap_angle_rad(float(heading_np[j] - heading_np[i]))

            ttc = compute_ttc_simple(
                pos_np[i, 0], pos_np[i, 1], vx_np[i], vy_np[i],
                pos_np[j, 0], pos_np[j, 1], vx_np[j], vy_np[j],
                collision_radius=self.collision_radius,
                max_ttc=self.max_ttc if self.max_ttc is not None else 10.0,
            )

            if self.max_ttc is not None and ttc > self.max_ttc:
                continue

            same_class = 1.0 if class_list[i] == class_list[j] else 0.0

            edge_feat_ij = [
                dx,
                dy,
                dist,
                dvx,
                dvy,
                rel_speed,
                closing_speed,
                math.sin(bearing),
                math.cos(bearing),
                math.sin(rel_heading),
                math.cos(rel_heading),
                ttc,
                same_class,
                float(size_np[i]),
                float(size_np[j]),
            ]
            src_nodes.append(i)
            dst_nodes.append(j)
            edge_rows.append(edge_feat_ij)

            if self.directed:
                dx_r = -dx
                dy_r = -dy
                dvx_r = -dvx
                dvy_r = -dvy
                bearing_r = math.atan2(dy_r, dx_r)
                rel_heading_r = wrap_angle_rad(float(heading_np[i] - heading_np[j]))
                closing_speed_r = -((dvx_r) * (dx_r / dist) + (dvy_r) * (dy_r / dist))

                edge_feat_ji = [
                    dx_r,
                    dy_r,
                    dist,
                    dvx_r,
                    dvy_r,
                    rel_speed,
                    closing_speed_r,
                    math.sin(bearing_r),
                    math.cos(bearing_r),
                    math.sin(rel_heading_r),
                    math.cos(rel_heading_r),
                    ttc,
                    same_class,
                    float(size_np[j]),
                    float(size_np[i]),
                ]
                src_nodes.append(j)
                dst_nodes.append(i)
                edge_rows.append(edge_feat_ji)

        edge_index = (
            torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
            if src_nodes
            else torch.empty((2, 0), dtype=torch.long)
        )

        edge_np = np.asarray(edge_rows, dtype=np.float32) if edge_rows else np.empty((0, 15), dtype=np.float32)

        if self.normalize and self.stats is not None:
            x_np = self._normalize_array(x_np, self.stats.node_mean, self.stats.node_std)
            edge_np = self._normalize_array(edge_np, self.stats.edge_mean, self.stats.edge_std)

        x = torch.tensor(x_np, dtype=torch.float32)
        edge_attr = torch.tensor(edge_np, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(pos_np, dtype=torch.float32),
            num_nodes=x.shape[0],
        )
        data.track_ids = torch.tensor(track_ids, dtype=torch.long)
        data.frame_id = int(frame_id if frame_id is not None else (frame_ids[0] if frame_ids else 0))
        if video_id is not None:
            data.video_id = video_id
        data.actor_class_index = torch.tensor(
            [self.actor_classes.index(c) if c in self.actor_classes else self.actor_classes.index("other") for c in class_list],
            dtype=torch.long,
        )
        return data

    def build_graph_for_frame(
        self,
        actors: list[dict[str, Any]],
        frame_id: int | None = None,
        video_id: str | None = None,
    ) -> dict[str, Any]:
        data = self.build_pyg_data(actors=actors, frame_id=frame_id, video_id=video_id)

        nodes = []
        for i in range(data.num_nodes):
            feat = data.x[i].tolist()
            cls_idx = int(data.actor_class_index[i].item())
            nodes.append(
                {
                    "node_id": i,
                    "track_id": int(data.track_ids[i].item()),
                    "actor_class": self.actor_classes[cls_idx],
                    "features": feat,
                    "x": float(data.pos[i, 0].item()),
                    "y": float(data.pos[i, 1].item()),
                }
            )

        edges = []
        for k in range(data.edge_index.shape[1]):
            src = int(data.edge_index[0, k].item())
            dst = int(data.edge_index[1, k].item())
            ef = data.edge_attr[k].tolist()
            edges.append(
                {
                    "src_node": src,
                    "dst_node": dst,
                    "dx": ef[0],
                    "dy": ef[1],
                    "distance": ef[2],
                    "dvx": ef[3],
                    "dvy": ef[4],
                    "relative_speed": ef[5],
                    "closing_speed": ef[6],
                    "bearing_sin": ef[7],
                    "bearing_cos": ef[8],
                    "rel_heading_sin": ef[9],
                    "rel_heading_cos": ef[10],
                    "ttc": ef[11],
                    "same_class": bool(round(ef[12])),
                    "size_src": ef[13],
                    "size_dst": ef[14],
                }
            )

        return {
            "frame_id": int(data.frame_id),
            "video_id": getattr(data, "video_id", None),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index.shape[1]),
            "nodes": nodes,
            "edges": edges,
            "radius": self.radius,
            "directed": self.directed,
        }

    def validate_graph(self, data: Data) -> dict[str, bool]:
        checks = {
            "has_nodes": int(data.num_nodes) > 0,
            "has_edges": data.edge_index.shape[1] > 0,
            "finite_node_features": bool(torch.isfinite(data.x).all()) if data.x.numel() else True,
            "finite_edge_features": bool(torch.isfinite(data.edge_attr).all()) if data.edge_attr.numel() else True,
            "valid_edge_indices": True,
            "no_self_loops": True,
        }

        if data.edge_index.numel():
            checks["valid_edge_indices"] = bool((data.edge_index >= 0).all() and (data.edge_index < data.num_nodes).all())
            checks["no_self_loops"] = not bool((data.edge_index[0] == data.edge_index[1]).any())

        return checks

    def _find_candidate_pairs(self, pos_np: np.ndarray) -> list[tuple[int, int]]:
        n = len(pos_np)
        if n == 0:
            return []

        if self.include_self_loops:
            pairs = [(i, i) for i in range(n)]
        else:
            pairs = []

        if n == 1:
            return pairs

        if self.use_kdtree and cKDTree is not None:
            search_radius = self._max_search_radius()
            tree = cKDTree(pos_np)
            undirected_pairs = sorted(tree.query_pairs(r=search_radius))
            pairs.extend((int(i), int(j)) for i, j in undirected_pairs)
            return pairs

        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((i, j))
        return pairs

    def _max_search_radius(self) -> float:
        max_pair_radius = max(self.pair_radii.values()) if self.pair_radii else self.radius
        base = max(self.radius, max_pair_radius)
        if self.use_velocity_adaptive_radius:
            base += 20.0 * self.velocity_radius_scale
        return base

    @staticmethod
    def _normalize_array(
        arr: np.ndarray,
        mean: np.ndarray | None,
        std: np.ndarray | None,
    ) -> np.ndarray:
        if arr.size == 0 or mean is None or std is None:
            return arr
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        std = np.where(std < 1e-6, 1.0, std)
        return (arr - mean) / std


def compute_feature_stats(graphs: Iterable[Data]) -> FeatureStats:
    node_blocks: list[np.ndarray] = []
    edge_blocks: list[np.ndarray] = []

    for g in graphs:
        if getattr(g, "x", None) is not None and g.x.numel():
            node_blocks.append(g.x.detach().cpu().numpy())
        if getattr(g, "edge_attr", None) is not None and g.edge_attr.numel():
            edge_blocks.append(g.edge_attr.detach().cpu().numpy())

    node_mean = node_std = edge_mean = edge_std = None

    if node_blocks:
        node_concat = np.concatenate(node_blocks, axis=0)
        node_mean = node_concat.mean(axis=0).astype(np.float32)
        node_std = node_concat.std(axis=0).astype(np.float32)

    if edge_blocks:
        edge_concat = np.concatenate(edge_blocks, axis=0)
        edge_mean = edge_concat.mean(axis=0).astype(np.float32)
        edge_std = edge_concat.std(axis=0).astype(np.float32)

    return FeatureStats(
        node_mean=node_mean,
        node_std=node_std,
        edge_mean=edge_mean,
        edge_std=edge_std,
    )


def build_graph_for_frame(
    actors: list[dict[str, Any]],
    radius: float = 6.0,
    actor_classes: list[str] | None = None,
    directed: bool = True,
) -> dict[str, Any]:
    builder = GraphBuilder(
        radius=radius,
        directed=directed,
        actor_classes=actor_classes,
    )
    return builder.build_graph_for_frame(actors=actors)


def build_interaction_graph(
    actors: Iterable[dict[str, Any]],
    radius: float = 6.0,
    actor_classes: list[str] | None = None,
    directed: bool = True,
) -> dict[str, Any]:
    return build_graph_for_frame(
        actors=list(actors),
        radius=radius,
        actor_classes=actor_classes,
        directed=directed,
    )


def build_pyg_graph_for_frame(
    actors: list[dict[str, Any]],
    radius: float = 6.0,
    actor_classes: list[str] | None = None,
    directed: bool = True,
    frame_id: int | None = None,
    video_id: str | None = None,
) -> Data:
    builder = GraphBuilder(
        radius=radius,
        directed=directed,
        actor_classes=actor_classes,
    )
    return builder.build_pyg_data(actors=actors, frame_id=frame_id, video_id=video_id)
