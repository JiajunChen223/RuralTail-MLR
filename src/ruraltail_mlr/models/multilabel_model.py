from __future__ import annotations

import torch
from torch import nn

from .classifier_heads import LinearHead


class RuralTailMLRModel(nn.Module):
    """Backbone plus a linear multi-label classification head."""

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = LinearHead(feature_dim, num_classes, dropout=dropout)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.backbone(image))
