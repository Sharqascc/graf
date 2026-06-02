from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class GraphSampleDataset:
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
    def from_jsonl(cls, path: str | Path) -> "GraphSampleDataset":
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
