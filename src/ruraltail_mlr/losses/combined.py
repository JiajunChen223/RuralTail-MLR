from __future__ import annotations

import torch
from torch import nn


class CombinedLoss(nn.Module):
    def __init__(self, cls_loss: nn.Module) -> None:
        super().__init__()
        self.cls_loss = cls_loss

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux: dict | None = None) -> dict[str, torch.Tensor]:
        loss_pack = self.cls_loss(logits, targets)
        if isinstance(loss_pack, dict):
            out = dict(loss_pack)
            out.setdefault("loss_cls", out["loss"].detach())
            return out
        return {"loss": loss_pack, "loss_cls": loss_pack.detach()}
