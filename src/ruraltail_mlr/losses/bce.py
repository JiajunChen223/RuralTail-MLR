from __future__ import annotations

import torch
from torch import nn


class BCEWithLogitsLoss(nn.Module):
    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        self.loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss(logits, targets.float())
