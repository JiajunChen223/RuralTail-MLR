from __future__ import annotations

import torch
from torch import nn


class LinearHead(nn.Module):
    def __init__(self, feature_dim: int, num_classes: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feature_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(features))
