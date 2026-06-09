from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn

from .asl import AsymmetricLoss


def _get(cfg, key: str, default=None):
    if cfg is None:
        return default
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _group_indices_from_frequency_csv(path: str | Path) -> dict[str, list[int]]:
    df = pd.read_csv(path).sort_values("class_idx")
    if "class_idx" not in df.columns:
        raise ValueError("frequency_csv must contain class_idx for tail-aware SoftF1.")
    if "group" not in df.columns:
        return {"default": [int(x) for x in df["class_idx"].tolist()]}
    groups: dict[str, list[int]] = {}
    for group_name, group_df in df.groupby("group", sort=False):
        groups[str(group_name)] = [int(x) for x in group_df["class_idx"].tolist()]
    return groups


def _logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(torch.logit(torch.tensor(p)).item())


class TailAwareSoftF1Term(nn.Module):
    """Mini-batch stable frequency-grouped SoftF1 and optional tail recall.

    Classes with no positive targets in the current mini-batch are skipped for
    that batch's SoftF1 estimate; the train-only H/M/T groups still define the
    reference thresholds and group weights.
    """

    def __init__(
        self,
        frequency_csv: str | Path = "data/processed/class_frequency.csv",
        lambda_soft_f1: float = 0.2,
        lambda_tail_recall: float = 0.05,
        soft_temperature: float = 0.5,
        threshold_by_group: dict | None = None,
        group_weights: dict | None = None,
        tail_recall_target: float = 0.65,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.group_indices = _group_indices_from_frequency_csv(frequency_csv)
        threshold_by_group = threshold_by_group or {
            "head": 0.5,
            "medium": 0.35,
            "tail": 0.25,
            "default": 0.5,
        }
        group_weights = group_weights or {
            "head": 1.0,
            "medium": 1.25,
            "tail": 1.5,
            "default": 1.0,
        }
        num_classes = max(idx for idxs in self.group_indices.values() for idx in idxs) + 1
        threshold_logits = torch.zeros(num_classes, dtype=torch.float32)
        for group_name, idxs in self.group_indices.items():
            threshold = threshold_by_group.get(group_name, threshold_by_group.get("default", 0.5))
            threshold_logits[idxs] = _logit(float(threshold))
        self.register_buffer("threshold_logits", threshold_logits)
        self.group_weights = {str(k): float(v) for k, v in group_weights.items()}
        self.lambda_soft_f1 = float(lambda_soft_f1)
        self.lambda_tail_recall = float(lambda_tail_recall)
        self.soft_temperature = max(float(soft_temperature), 1e-3)
        self.tail_recall_target = float(tail_recall_target)
        self.eps = float(eps)

    def _soft_group_terms(self, logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        targets = targets.float()
        threshold_logits = self.threshold_logits.to(device=logits.device, dtype=logits.dtype)
        soft_pred = torch.sigmoid((logits - threshold_logits[None, :]) / self.soft_temperature)
        losses = []
        weights = []
        tail_recall_losses = []
        for group_name, idxs in self.group_indices.items():
            idx_tensor = torch.tensor(idxs, device=logits.device, dtype=torch.long)
            group_targets = targets.index_select(dim=1, index=idx_tensor)
            positive_mask = group_targets.sum(dim=0) > 0
            if not bool(positive_mask.any()):
                continue
            group_pred = soft_pred.index_select(dim=1, index=idx_tensor)[:, positive_mask]
            group_targets = group_targets[:, positive_mask]
            tp = (group_pred * group_targets).sum(dim=0)
            fp = (group_pred * (1.0 - group_targets)).sum(dim=0)
            fn = ((1.0 - group_pred) * group_targets).sum(dim=0)
            soft_f1 = (2.0 * tp + self.eps) / (2.0 * tp + fp + fn + self.eps)
            losses.append(1.0 - soft_f1.mean())
            weights.append(self.group_weights.get(group_name, self.group_weights.get("default", 1.0)))
            if group_name == "tail" and self.lambda_tail_recall > 0:
                recall = (tp + self.eps) / (tp + fn + self.eps)
                tail_recall_losses.append(F.relu(self.tail_recall_target - recall.mean()).pow(2))
        if not losses:
            zero = logits.sum() * 0.0
            return zero, zero
        weight_tensor = torch.tensor(weights, device=logits.device, dtype=logits.dtype)
        loss_tensor = torch.stack(losses)
        soft_f1_loss = (loss_tensor * weight_tensor).sum() / weight_tensor.sum().clamp(min=self.eps)
        if tail_recall_losses:
            tail_recall_loss = torch.stack(tail_recall_losses).mean()
        else:
            tail_recall_loss = logits.sum() * 0.0
        return soft_f1_loss, tail_recall_loss

class ASLWithTSoftF1Loss(TailAwareSoftF1Term):
    """ASL plus the same frequency-grouped soft-F1 and tail recall terms."""

    def __init__(
        self,
        frequency_csv: str | Path = "data/processed/class_frequency.csv",
        lambda_soft_f1: float = 0.2,
        lambda_tail_recall: float = 0.05,
        soft_temperature: float = 0.5,
        threshold_by_group: dict | None = None,
        group_weights: dict | None = None,
        tail_recall_target: float = 0.65,
        gamma_pos: float = 0.0,
        gamma_neg: float = 4.0,
        clip: float = 0.05,
        eps: float = 1e-8,
    ) -> None:
        super().__init__(
            frequency_csv=frequency_csv,
            lambda_soft_f1=lambda_soft_f1,
            lambda_tail_recall=lambda_tail_recall,
            soft_temperature=soft_temperature,
            threshold_by_group=threshold_by_group,
            group_weights=group_weights,
            tail_recall_target=tail_recall_target,
            eps=eps,
        )
        self.asl = AsymmetricLoss(gamma_pos=gamma_pos, gamma_neg=gamma_neg, clip=clip, eps=eps)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> dict[str, torch.Tensor]:
        targets = targets.float()
        asl = self.asl(logits, targets)
        soft_f1_loss, tail_recall_loss = self._soft_group_terms(logits, targets)
        loss = asl + self.lambda_soft_f1 * soft_f1_loss + self.lambda_tail_recall * tail_recall_loss
        return {
            "loss": loss,
            "loss_asl": asl.detach(),
            "loss_soft_f1": soft_f1_loss.detach(),
            "loss_tail_recall": tail_recall_loss.detach(),
        }
