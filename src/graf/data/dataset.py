from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset


class PtGraphDataset(Dataset):
    """
    PyG-native dataset backed by individual .pt files.

    Each .pt file must contain a single torch_geometric.data.Data object.
    """

    def __init__(
        self,
        root: str | Path,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        recursive: bool = False,
    ) -> None:
        self.root_path = Path(root).expanduser().resolve()
        self.recursive = recursive
        super().__init__(str(self.root_path), transform, pre_transform, pre_filter)

        if not self.root_path.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root_path}")

        pattern = "**/*.pt" if self.recursive else "*.pt"
        self._files = sorted(self.root_path.glob(pattern))

    @property
    def raw_file_names(self) -> list[str]:
        return []

    @property
    def processed_file_names(self) -> list[str]:
        return [p.name for p in self._files]

    def len(self) -> int:
        return len(self._files)

    def get(self, idx: int) -> Data:
        path = self._files[idx]
        data = torch.load(path, map_location="cpu", weights_only=False)
        if not isinstance(data, Data):
            raise TypeError(f"Expected a PyG Data object in {path}, got {type(data)!r}")
        return data

    def file_path(self, idx: int) -> Path:
        return self._files[idx]
