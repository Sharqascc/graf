from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from graf.graph.pyg_export import to_pyg_data


class GraphSampleDataset:
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        graph = sample["graph"]
        label = sample.get("label", 0)
        data = to_pyg_data(graph)
        try:
            import torch
            data.y = torch.tensor([label], dtype=torch.float32)
        except Exception:
            data["y"] = [float(label)]
        return data

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "GraphSampleDataset":
        path = Path(path)
        samples = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                samples.append(json.loads(line))
        return cls(samples)
