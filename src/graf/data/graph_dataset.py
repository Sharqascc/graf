from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from torch.utils.data import Dataset
from torch_geometric.data import Data

from graf.data.dataset import PtGraphDataset
from graf.graph.temporal import build_temporal_window_graph


@dataclass
class LegacyGraphSampleDataset:
    samples: list[dict[str, Any]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.samples[idx]

    @staticmethod
    def _normalize_sample(row: dict[str, Any]) -> dict[str, Any]:
        graph = row.get("graph", row)

        if not isinstance(graph, dict):
            graph = {}

        sample = {
            "nodes": graph.get("nodes", []),
            "edges": graph.get("edges", []),
            "label": row.get("label", graph.get("label", 0)),
            "sample_id": row.get("sample_id", graph.get("sample_id")),
            "site_id": row.get("site_id", graph.get("site_id")),
            "video_id": row.get("video_id", graph.get("video_id")),
            "window_id": row.get("window_id", graph.get("window_id")),
        }

        for key, value in graph.items():
            if key not in sample:
                sample[key] = value

        for key, value in row.items():
            if key not in sample and key != "graph":
                sample[key] = value

        return sample

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "LegacyGraphSampleDataset":
        path = Path(path)
        samples: list[dict[str, Any]] = []

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                samples.append(cls._normalize_sample(row))

        return cls(samples)


class SpatioTemporalWindowDataset(Dataset):
    """
    Build rolling spatio-temporal window graphs from per-frame .pt PyG Data files.

    Each item is a single PyG Data object representing one temporal window.
    """

    def __init__(
        self,
        graph_dir: str | Path,
        window_size: int = 8,
        stride: int = 1,
        recursive: bool = False,
        connect_bidirectional: bool = True,
        temporal_self_only: bool = False,
    ) -> None:
        self.base_dataset = PtGraphDataset(graph_dir, recursive=recursive)
        self.window_size = int(window_size)
        self.stride = int(stride)
        self.connect_bidirectional = connect_bidirectional
        self.temporal_self_only = temporal_self_only

        if self.window_size <= 0:
            raise ValueError("window_size must be positive")
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        self.windows = self._build_windows()

    def _build_windows(self) -> list[list[int]]:
        groups: dict[str, list[tuple[int, int]]] = {}

        for idx in range(len(self.base_dataset)):
            data = self.base_dataset.get(idx)
            video_id = str(getattr(data, "video_id", "unknown"))
            frame_id = int(getattr(data, "frame_id", idx))
            groups.setdefault(video_id, []).append((frame_id, idx))

        windows: list[list[int]] = []
        for _, items in groups.items():
            items.sort(key=lambda x: x[0])
            ordered_indices = [idx for _, idx in items]

            if len(ordered_indices) < self.window_size:
                continue

            for start in range(0, len(ordered_indices) - self.window_size + 1, self.stride):
                windows.append(ordered_indices[start : start + self.window_size])

        return windows

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Data:
        frame_indices = self.windows[idx]
        frames = [self.base_dataset.get(i) for i in frame_indices]

        data = build_temporal_window_graph(
            frames,
            connect_bidirectional=self.connect_bidirectional,
            temporal_self_only=self.temporal_self_only,
        )
        data.window_index = idx
        return data
