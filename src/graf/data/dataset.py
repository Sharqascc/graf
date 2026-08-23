from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset

import torch_geometric.data.data as pyg_data
import torch_geometric.data.storage as pyg_storage
torch.serialization.add_safe_globals([
    Data,
    pyg_data.DataEdgeAttr,
    pyg_data.DataTensorAttr,
    pyg_storage.GlobalStorage,
    pyg_storage.BaseStorage,
    pyg_storage.NodeStorage,
    pyg_storage.EdgeStorage,
])



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
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_path}"
            )

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
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            raise IOError(f"Failed to load graph from {path}: {e}") from e
        if not isinstance(data, Data):
            raise TypeError(f"Expected a PyG Data object in {path}, got {type(data)!r}")
        return data

    def file_path(self, idx: int) -> Path:
        return self._files[idx]
