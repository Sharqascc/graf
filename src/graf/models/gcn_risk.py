from __future__ import annotations

from typing import Any, Literal


def has_torch_geometric() -> bool:
    try:
        import torch  # noqa: F401
        import torch_geometric  # noqa: F401
        return True
    except Exception:
        return False


class SimpleGCNRiskModel:
    """Placeholder model returned when torch or torch_geometric is unavailable."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = 32,
        num_layers: int = 2,
        dropout: float = 0.0,
        pool: str = "mean",
    ) -> None:
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.dropout = dropout
        self.pool = pool

    def __repr__(self) -> str:
        return (
            "SimpleGCNRiskModel("
            f"in_channels={self.in_channels}, "
            f"hidden_channels={self.hidden_channels}, "
            f"num_layers={self.num_layers}, "
            f"dropout={self.dropout}, "
            f"pool='{self.pool}')"
        )

    def is_torch_ready(self) -> bool:
        return has_torch_geometric()

    def forward(self, data: Any) -> Any:
        raise RuntimeError(
            "GCNRiskModel requires torch and torch_geometric to be installed."
        )

    __call__ = forward


try:
    import torch
    from torch import nn
    from torch_geometric.nn import (
        GCNConv,
        global_add_pool,
        global_max_pool,
        global_mean_pool,
    )

    _POOLERS = {
        "mean": global_mean_pool,
        "max": global_max_pool,
        "add": global_add_pool,
    }

    class GCNRiskModel(nn.Module):
        """Graph-level binary risk model producing logits for BCEWithLogitsLoss."""

        def __init__(
            self,
            in_channels: int,
            hidden_channels: int = 32,
            num_layers: int = 2,
            dropout: float = 0.0,
            pool: Literal["mean", "max", "add"] = "mean",
        ) -> None:
            super().__init__()

            if num_layers < 2:
                raise ValueError(f"num_layers must be >= 2, got {num_layers}")
            if pool not in _POOLERS:
                raise ValueError(f"Unsupported pool='{pool}'")

            self.in_channels = in_channels
            self.hidden_channels = hidden_channels
            self.num_layers = num_layers
            self.dropout = dropout
            self.pool = pool

            self.convs = nn.ModuleList()
            self.convs.append(GCNConv(in_channels, hidden_channels))
            for _ in range(num_layers - 1):
                self.convs.append(GCNConv(hidden_channels, hidden_channels))

            self.activation = nn.ReLU()
            self.dropout_layer = nn.Dropout(dropout)
            self.head = nn.Linear(hidden_channels, 1)
            self.pool_fn = _POOLERS[pool]

        def forward(self, data: Any) -> torch.Tensor:
            x, edge_index = data.x, data.edge_index
            batch = getattr(data, "batch", None)

            if batch is None:
                batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

            for conv in self.convs:
                x = conv(x, edge_index)
                x = self.activation(x)
                x = self.dropout_layer(x)

            x = self.pool_fn(x, batch)
            return self.head(x).view(-1)

        def __repr__(self) -> str:
            return (
                "GCNRiskModel("
                f"in_channels={self.in_channels}, "
                f"hidden_channels={self.hidden_channels}, "
                f"num_layers={self.num_layers}, "
                f"dropout={self.dropout}, "
                f"pool='{self.pool}')"
            )

except Exception:
    GCNRiskModel = None


def build_model(
    in_channels: int,
    hidden_channels: int = 32,
    num_layers: int = 2,
    dropout: float = 0.0,
    pool: str = "mean",
) -> Any:
    if GCNRiskModel is None:
        return SimpleGCNRiskModel(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            pool=pool,
        )

    return GCNRiskModel(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        pool=pool,
    )
