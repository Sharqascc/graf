from __future__ import annotations

from typing import Any


class SimpleGCNRiskModel:
    def __init__(self, in_channels: int, hidden_channels: int = 32):
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

    def __repr__(self) -> str:
        return f"SimpleGCNRiskModel(in_channels={self.in_channels}, hidden_channels={self.hidden_channels})"

    def is_torch_ready(self) -> bool:
        try:
            import torch  # noqa: F401
            import torch_geometric  # noqa: F401
            return True
        except Exception:
            return False


def build_model(in_channels: int, hidden_channels: int = 32) -> Any:
    try:
        import torch
        from torch import nn
        from torch_geometric.nn import GCNConv, global_mean_pool
    except Exception:
        return SimpleGCNRiskModel(in_channels=in_channels, hidden_channels=hidden_channels)

    class GCNRiskModel(nn.Module):
        def __init__(self, in_channels: int, hidden_channels: int):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.head = nn.Linear(hidden_channels, 1)

        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)
            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index).relu()
            x = global_mean_pool(x, batch)
            return self.head(x).view(-1)

    return GCNRiskModel(in_channels=in_channels, hidden_channels=hidden_channels)
